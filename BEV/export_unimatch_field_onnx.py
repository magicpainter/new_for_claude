#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export UniMatch-V2 (DPT) checkpoint to ONNX with raw logits output.

- Input : 1x3x1200x1920 RGB float32/float16/bfloat16 in [0,1]
- Model : auto-detects DPT (DINOv2) variant (small/base/large/giant) from checkpoint
- Resize: **downscale to (inner_h, inner_w) inside graph, forward, then upscale back to (H,W)**
- Output: 1xCx1200x1920 logits (不做 argmax；后处理在推理脚本中进行)
- Opset : 17
"""

# ===================== 必须最先：禁用 xFormers / Flash-Attn =====================
import os as _os
_os.environ.setdefault("XFORMERS_DISABLED", "1")               # DINOv2 官方识别
_os.environ.setdefault("XFORMERS_DISABLE_FLASH_ATTN", "1")     # 旧版 xformers 识别
# ===============================================================================

import argparse
import math
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# 优先使用新版 SDPA 上下文（若不可用则降级为空上下文）
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend  # PyTorch ≥ 2.3
except Exception:  # 兼容旧版 Torch
    import contextlib as _ctx
    def sdpa_kernel(_):  # type: ignore
        return _ctx.nullcontext()
    class SDPBackend:  # type: ignore
        MATH = None

# 项目内导入（现在环境变量已生效）
from model.semseg.dpt import DPT
import model.backbone.dinov2_layers.attention as dino_attn

# =============== 二次保险：全局禁用 flash/mem-efficent SDPA，启用 MATH ===============
try:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass
# ===============================================================================

# ----------------------------- utils -----------------------------

def strip_module_prefix(state_dict):
    """Remove 'module.' (DDP) prefix if present."""
    return {k.replace('module.', ''): v for k, v in state_dict.items()}

def load_ckpt(ckpt_path, device="cpu"):
    """
    Robust loader for PyTorch≥2.6 (weights_only=True 默认) 与更早版本。
    你自己的权重来源可信，这里显式 weights_only=False。
    """
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    except Exception:
        try:
            import numpy as np
            from torch.serialization import add_safe_globals
            add_safe_globals([np.core.multiarray.scalar])
            ckpt = torch.load(ckpt_path, map_location=device)
        except Exception:
            raise

    # 优先用 EMA（若存在），否则用 model；否则当成裸 state_dict
    if isinstance(ckpt, dict):
        if "model_ema" in ckpt and isinstance(ckpt["model_ema"], dict):
            sd = strip_module_prefix(ckpt["model_ema"])
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            sd = strip_module_prefix(ckpt["model"])
        else:
            sd = strip_module_prefix(ckpt)
    else:
        raise RuntimeError(f"Unexpected checkpoint format: {type(ckpt)}")
    return sd

def build_dpt_for_variant(variant: str, nclass: int):
    """Return DPT model for given variant name."""
    model_configs = {
        "small": {'encoder_size': 'small', 'features': 64,  'out_channels': [48, 96, 192, 384]},
        "base":  {'encoder_size': 'base',  'features': 128, 'out_channels': [96, 192, 384, 768]},
        "large": {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        "giant": {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
    }
    cfg = model_configs[variant]
    return DPT(**{**cfg, 'nclass': nclass})

def autodetect_variant(state_dict):
    """
    Detect DINOv2 variant by embed_dim in checkpoint:
      384→small, 768→base, 1024→large, 1536→giant
    """
    dim = None
    if "backbone.cls_token" in state_dict:
        dim = int(state_dict["backbone.cls_token"].shape[-1])
    elif "backbone.pos_embed" in state_dict:
        dim = int(state_dict["backbone.pos_embed"].shape[-1])
    elif "backbone.patch_embed.proj.weight" in state_dict:
        dim = int(state_dict["backbone.patch_embed.proj.weight"].shape[0])

    mapping = {384: "small", 768: "base", 1024: "large", 1536: "giant"}
    variant = mapping.get(dim, "base")
    print(f"[auto] detected embed_dim={dim} → variant='{variant}'")
    return variant

def load_state_dict_shape_safe(model: nn.Module, sd: dict):
    """只加载 shape 兼容的权重（最大限度容错）。"""
    msd = model.state_dict()
    compat = {k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}
    missing = sorted(set(msd.keys()) - set(compat.keys()))
    unexpected = sorted(set(sd.keys()) - set(compat.keys()))
    model.load_state_dict(compat, strict=False)
    print(f"[load] loaded {len(compat)} tensors, missing={len(missing)}, unexpected={len(unexpected)}")

# ------------------------- ONNX-safe 注意力猴补 ------------------------

def _onnx_safe_me_attention(q, k, v, attn_bias=None, p: float = 0.0, scale=None, **kwargs):
    """
    纯 PyTorch 实现的 scaled dot-product attention，形状与常见 xFormers 调用保持一致：
      输入 q/k/v: (B, M, H, K) → 内部转为 (B, H, M, K)
      输出      : (B, M, H, K)
    仅用于导出（不追求最优性能），避免任何自定义算子域。
    """
    q_ = q.permute(0, 2, 1, 3)
    k_ = k.permute(0, 2, 1, 3)
    v_ = v.permute(0, 2, 1, 3)

    Kdim = q_.shape[-1]
    scale_t = (1.0 / math.sqrt(float(Kdim))) if scale is None else float(scale)

    attn = torch.matmul(q_, k_.transpose(-2, -1)) * scale_t  # (B,H,M,M)
    if attn_bias is not None:
        attn = attn + attn_bias
    attn = torch.softmax(attn, dim=-1)
    out = torch.matmul(attn, v_)                              # (B,H,M,K)
    return out.permute(0, 2, 1, 3)                            # (B,M,H,K)

# 无条件替换（导出期专用）
if hasattr(dino_attn, "memory_efficient_attention"):
    dino_attn.memory_efficient_attention = _onnx_safe_me_attention
    print("[patch] dino_attn.memory_efficient_attention -> ONNX-safe implementation")

# ------------------------- deploy wrapper ------------------------

class DeploySeg(nn.Module):
    """
    Wrap DPT with:
      - **downsample to (inner_h, inner_w)**
      - forward at inner size
      - **upsample logits back to (H, W)**
      - return raw logits (float), no argmax
    """
    def __init__(self, core: nn.Module, H: int = 1200, W: int = 1920,
                 inner_h: int = 560, inner_w: int = 896):
        super().__init__()
        self.core = core
        self.H = int(H)
        self.W = int(W)
        self.inner_h = int(inner_h)
        self.inner_w = int(inner_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (1,3,H,W) float in [0,1], RGB
        return: (1,C,H,W) logits (dtype 与模型一致)
        """
        torch._assert(x.dim() == 4, "expect NCHW input")
        torch._assert(x.shape[0] == 1, "expect batch=1")
        torch._assert(x.shape[1] == 3, "expect 3-channel RGB")

        H, W = self.H, self.W
        x = torch.clamp(x, 0.0, 1.0)

        # 1) 固定降采样到 inner 尺寸（例如 560x896，建议 14 的倍数）
        x_small = F.interpolate(x, (self.inner_h, self.inner_w),
                                mode='bilinear', align_corners=True)

        # 2) 主网络在小分辨率进行
        with sdpa_kernel(SDPBackend.MATH):
            logits_small = self.core(x_small)  # (1, C, inner_h, inner_w)

        # 3) 把 logits 升采样回外部尺寸 (H, W)
        if (self.inner_h, self.inner_w) != (H, W):
            logits = F.interpolate(logits_small, (H, W),
                                   mode='bilinear', align_corners=True)
        else:
            logits = logits_small
        return logits

# ------------------------------ main -----------------------------

def _require_onnx_installed():
    try:
        import onnx  # noqa: F401
    except ModuleNotFoundError:
        msg = (
            "[error] Python package 'onnx' is required for torch.onnx.export.\n"
            "Please install it in your current environment, e.g.:\n"
            "  pip install --upgrade onnx\n"
            "  # or\n"
            "  conda install -c conda-forge onnx\n"
        )
        print(msg, file=sys.stderr)
        sys.exit(1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default='exp/field/unimatch_v2_minibatch/dinov2_base/138_3052_saved/latest.pth', type=str,
                    help="path to latest.pth / best.pth")
    ap.add_argument("--output", default='exports/unimatch_field_1200x1920_logits_op17_280_448.onnx', type=str,
                    help="output .onnx path")
    ap.add_argument("--height", type=int, default=1200)
    ap.add_argument("--width",  type=int, default=1920)
    # >>> 新增：图内计算的固定尺寸（默认 560x896） <<<
    ap.add_argument("--inner_height", type=int, default=560) #was 560
    ap.add_argument("--inner_width",  type=int, default=896) #was 896

    ap.add_argument("--opset",  type=int, default=17)
    ap.add_argument("--force_variant", type=str, default="base",
                    choices=["", "small", "base", "large", "giant"],
                    help="留空自动探测；否则强制指定（small/base/large/giant）")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--precision", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    args = ap.parse_args()

    # -- device
    device = args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"[env] device={device}, precision={args.precision}, opset={args.opset}")
    print(f"[cfg] outer=({args.height},{args.width}), inner=({args.inner_height},{args.inner_width})")

    # -- load weights
    sd = load_ckpt(args.checkpoint, device="cpu")  # 先在 CPU 上安全加载

    # -- pick variant
    variant = (args.force_variant or "").strip() or autodetect_variant(sd)
    print(f"[info] using variant: {variant}")

    # -- build model and load weights (shape-safe)
    core = build_dpt_for_variant(variant, nclass=5)
    load_state_dict_shape_safe(core, sd)
    core.eval()

    # -- move to device & dtype
    if device == "cuda":
        if args.precision == "fp16":
            core = core.half().cuda()
        elif args.precision == "bf16":
            core = core.to(torch.bfloat16).cuda()
        else:
            core = core.float().cuda()
    else:
        core = core.float()  # CPU 路径

    deploy = DeploySeg(core, H=args.height, W=args.width,
                       inner_h=args.inner_height, inner_w=args.inner_width).eval()
    if device == "cuda":
        deploy = deploy.cuda()
        if args.precision == "fp16":
            deploy = deploy.half()
        elif args.precision == "bf16":
            deploy = deploy.to(torch.bfloat16)

    # -- dummy input（[0,1]）
    if device == "cuda":
        if args.precision == "fp16":
            dummy = torch.rand(1, 3, args.height, args.width, dtype=torch.float16, device="cuda")
        elif args.precision == "bf16":
            dummy = torch.rand(1, 3, args.height, args.width, dtype=torch.bfloat16, device="cuda")
        else:
            dummy = torch.rand(1, 3, args.height, args.width, dtype=torch.float32, device="cuda")
    else:
        dummy = torch.rand(1, 3, args.height, args.width, dtype=torch.float32)

    # -- 预跑一遍（可提早暴露其它问题）
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        _ = deploy(dummy)

    # -- export
    input_names = ["images"]
    output_names = ["logits"]  # (1, C, H, W) same dtype as model

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 你现在走的是 TorchScript 导出器路径，需要确保 onnx 包已安装
    _require_onnx_installed()

    # 明确打印当前导出器路径
    print("[info] exporter path = torchscript (dynamo=False)")

    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        torch.onnx.export(
            deploy,
            dummy,
            args.output,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=None,      # 固定 1x3xH xW
            # dynamo=False
        )

    print(f"[done] ONNX saved to: {args.output}")
    print(f"       input  : (1,3,{args.height},{args.width}) {dummy.dtype} on {device} in [0,1]")
    print(f"       output : (1,C,{args.height},{args.width}) logits (no argmax)")

if __name__ == "__main__":
    main()
