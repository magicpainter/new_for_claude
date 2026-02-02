import os
import hashlib
from getpass import getpass
from pathlib import Path


def set_gate_password(conf_path: str, *, confirm: bool = True, chmod_600: bool = True) -> None:
    """
    Set/update the upload gate password in a config file.

    Updates/creates:
      gate_salt = <hex>
      gate_sha256 = <hex sha256(salt_bytes + password_bytes)>

    Keeps other lines (like remote_dir) untouched.
    """
    p = Path(conf_path).expanduser()

    # Read existing lines (if any)
    lines = p.read_text(encoding="utf-8").splitlines() if p.exists() else []

    # Prompt for new password
    pw1 = getpass("Set NEW upload gate password: ")
    if confirm:
        pw2 = getpass("Confirm password: ")
        if pw1 != pw2:
            raise ValueError("Passwords do not match. Aborted.")

    # Generate salt + hash
    salt_bytes = os.urandom(8)
    gate_salt = salt_bytes.hex()
    gate_sha256 = hashlib.sha256(salt_bytes + pw1.encode("utf-8")).hexdigest()

    # Replace or append keys
    def upsert_key(key: str, value: str, existing_lines: list[str]) -> list[str]:
        prefix = key.strip() + " ="
        replaced = False
        new_lines = []
        for ln in existing_lines:
            if ln.strip().startswith("#") or "=" not in ln:
                new_lines.append(ln)
                continue
            k = ln.split("=", 1)[0].strip()
            if k == key:
                new_lines.append(f"{key} = {value}")
                replaced = True
            else:
                new_lines.append(ln)
        if not replaced:
            new_lines.append(f"{key} = {value}")
        return new_lines

    lines = upsert_key("gate_salt", gate_salt, lines)
    lines = upsert_key("gate_sha256", gate_sha256, lines)

    # Write atomically
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    tmp.replace(p)

    # Optional: lock down file permissions
    if chmod_600:
        try:
            os.chmod(p, 0o600)
        except PermissionError:
            pass

    print(f"Updated gate password in: {p}")


if __name__ == "__main__":
    set_gate_password("/home/mp/work/container_folder/backend_pipeline/upload_gate.conf")

