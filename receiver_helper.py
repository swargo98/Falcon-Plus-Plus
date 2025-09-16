# ──────────────────────────────────────────────────────────────────────────────
# receiver_helpers.py  – called on the SIDE THAT GENERATES THE LOGS
# ──────────────────────────────────────────────────────────────────────────────
import socket, struct, json, pathlib
from typing import Dict

def push_logs_to_sender(
    cfg: Dict,
    dest_host: str,
    dest_port: int = 6060,
    pattern: str = "*_ppo_*.csv",
    log_dir: str = pathlib.Path.cwd(),
    timeout: int = 30,
) -> None:
    """
    Stream every CSV matching *pattern* to the waiting peer, then exit.

    The listener decides the final filenames, so we just send the originals.

    Parameters
    ----------
    cfg : dict
        Your `configurations` dict (only needed if you want to log the
        model_version or raise customised errors).
    dest_host, dest_port : str | int
        Address of the listener.
    pattern : str
        Glob pattern of files to send (default '*_ppo_*.csv').
    log_dir : Path | str
        Directory to search for the log files.
    timeout : int
        Socket-connect timeout in seconds.
    """
    log_dir = pathlib.Path(log_dir)
    files   = [p for p in log_dir.glob(pattern) if p.is_file()]
    if not files:
        print("[pusher] no CSV logs found – nothing to send")
        return

    with socket.create_connection((dest_host, dest_port), timeout=timeout) as s:
        for path in files:
            header = json.dumps({"name": path.name,
                                 "size": path.stat().st_size}).encode()
            s.sendall(struct.pack(">I", len(header)))
            s.sendall(header)
            s.sendall(path.read_bytes())
            print(f"[pusher] sent {path.name}")
        s.sendall(struct.pack(">I", 0))   # EXIT token
    print("[pusher] all logs sent – done")