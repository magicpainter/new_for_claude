# !/usr/bin/env python3
import os, time, queue, threading
from datetime import datetime
from class_OD_offline_parallel_q import InferenceEngine
from GUI.GUI_q import run_gui
from GUI.GUI_q import PBOPool
import subprocess

def main():
    cfg = {
        "engine_file_path_shooting": "best_shoot_640_yolo11n_200e_map_84p6_0124_b4_fp16_nms.engine",
        "engine_file_path_dribble": "best_dribble_yolo11n_500e_map_98p2_0116_b2_fp16_nms.engine",
        "class_names": ["ball", "bib", "player"],
        "original_size": (1200, 1920), # original size (self.src_H, self.src_W)
        "score_threshold": 0.2,
        "batch_size": 4,
        "input_size": (640, 640),
        "output_folder": "./batch_results",
        "gpu_id": 0,
        "framerate": 60
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    left_out = os.path.join(cfg["output_folder"], f"left_{ts}")
    right_out = os.path.join(cfg["output_folder"], f"right_{ts}")
    #os.makedirs(left_out, exist_ok=True)
    #os.makedirs(right_out, exist_ok=True)


    error_q = queue.Queue()  # ´íÎó¶ÓÁÐ
    run_event = threading.Event()  # start and pause of pipeline
    cmd_queue = queue.Queue()  # GUI --> process command
    tex_queue = queue.Queue() # GUI -> backend (texture IDs)


    res_queue = queue.Queue(maxsize=400)  # process --> GUI return res

    ctx_q = queue.Queue(maxsize=1)  # ´æ·Å GL_CUDA_CTX µÄÏß³Ì°²È«¹ÜµÀ
    ctx_ready = threading.Event()  # GUI set() ÒÔºó£¬ÍÆÀíÏß³Ì²ÅÕæÕý¿ªÊ¼

    infer_frame_queue = queue.Queue(maxsize=10)
    lr_q = queue.Queue(maxsize=200)

    inf_q = queue.Queue()
    lres_q = queue.Queue()
    rres_q = queue.Queue()
    done_q = queue.Queue()
    gui2infer_queue = queue.Queue()

    # ºó´¦ÀíÍê³É±êÖ¾
    lpost_done = threading.Event()
    rpost_done = threading.Event()

    pbo_pool_size = 16
    pbo_pool = PBOPool(pbo_pool_size,
                       cfg["original_size"][1],  
                       cfg["original_size"][0])  

    # # Make sure nothing else is holding the cameras
    # subprocess.run(["fuser", "-v", "/dev/video0", "/dev/video1"])
    # subprocess.run(["lsof", "/dev/video0", "/dev/video1"])

    subprocess.run(
        ["v4l2-ctl", "-d", "/dev/video0", "--set-ctrl=horizontal_flip=1"],
        capture_output=True, text=True, check=True  # check=True ¡ú ·Ç 0 ·µ»ØÂë»áÅ×Òì³£
    )
    subprocess.run(
        ["v4l2-ctl", "-d", "/dev/video1", "--set-ctrl=horizontal_flip=1"],
        capture_output=True, text=True, check=True  # check=True ¡ú ·Ç 0 ·µ»ØÂë»áÅ×Òì³£
    )


    engine = InferenceEngine(cfg, lr_q, inf_q, done_q, error_q, cmd_queue, tex_queue, run_event,res_queue, ctx_q,
                             ctx_ready, gui2infer_queue, pbo_pool, infer_frame_queue)
    engine.start()

    run_gui(cmd_queue, tex_queue, res_queue, run_event, ctx_q, ctx_ready, cfg, gui2infer_queue, pbo_pool,
            infer_frame_queue, error_q=error_q, engine=engine)


    print("[Main] Dispatcher running...")
    loaders_done = 0
    inf_done = False
    batches = 0
    right_disp = 0

    try:
        while loaders_done < 2 or not inf_done:
            # ¼ì²é´íÎó¶ÓÁÐ
            if not error_q.empty():
                err_proc, err_msg = error_q.get()
                raise RuntimeError(f"Error in {err_proc}: {err_msg}")


            # ´¦ÀíÍÆÀí½á¹û
            while not inf_q.empty():
                batch = inf_q.get_nowait()
                if batch is None:
                    inf_done = True
                    print("[Main] Inference sentinel received")
                    continue
                batches += 1
                print(f"[Main] Dispatch batch#{batches} -> {len(batch)} imgs")
                for res in batch:
                    (lres_q if res['side'] == 'left' else rres_q).put(res)
                    if res['side'] == 'right':
                        right_disp += 1

            # ´¦Àí¼ÓÔØÆ÷Íê³ÉÐÅºÅ
            while not done_q.empty():
                msg = done_q.get_nowait()
                if msg in ("left", "right"):
                    loaders_done += 1
                    print(f"[Main] {msg} loader done {loaders_done}/2")
                elif msg == "inference":
                    inf_done = True

            time.sleep(0.01)

        # ·¢ËÍÖÕÖ¹ÐÅºÅ¸øºó´¦ÀíÆ÷
        lres_q.put(None)
        rres_q.put(None)

        # µÈ´ýºó´¦ÀíÍê³É
        print("[Main] Waiting for post-processors...")
        lpost_done.wait()
        rpost_done.wait()

    except Exception as e:
        print(f"[Main] Critical error: {str(e)}")
        # ·¢ËÍÖÕÖ¹ÐÅºÅ¸øËùÓÐ¶ÓÁÐ
        for q in [lr_q, inf_q, lres_q, rres_q]:
            while not q.empty(): q.get()
            q.put("TERMINATE")  # ÌØÊâÖÕÖ¹ÐÅºÅ

    finally:

        time.sleep(5) #sleep for 30 s

        cmd_queue.put({"cmd": "shutdown"})
        #gui_thread.join()
        engine.join(timeout=5)

        print(f"[Main] Finished - total batches {batches}")
        for q in (lr_q, inf_q, lres_q, rres_q):
            q.close()
            q.join_thread()



if __name__ == "__main__":
    t0 = time.time()

    main()
    print(f"[Main] wall-time {time.time() - t0:.2f}s")
