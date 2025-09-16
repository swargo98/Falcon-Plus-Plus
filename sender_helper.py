# ──────────────────────────────────────────────────────────────────────────────
# sender_helpers.py  – called on the SIDE THAT NEEDS THE LOGS
# ──────────────────────────────────────────────────────────────────────────────
import socket, struct, pathlib, json, threading
from typing import Dict, Optional

def start_log_listener(
    cfg: Dict,
    host: str = "0.0.0.0",
    port: int = 6060,
    num_workers: int = 1,
    background: bool = False,
) -> Optional[threading.Thread]:
    """
    Wait for a single peer to PUSH its CSV logs, then save/rename them locally.

    Parameters
    ----------
    cfg : dict
        Your existing `configurations` dict – must contain `model_version`
        and optionally `log_dir`.
    host, port : str | int
        Interface & port to bind.  Use 0.0.0.0 to listen on all IPv4 IFs.
    num_workers : int
        Passed to `socket.listen()`.  Keep 1 unless you expect >1 pusher.
    background : bool
        If True, run listener in a daemon thread and return it.  If False,
        the call blocks until all logs arrive.

    Returns
    -------
    Optional[threading.Thread]
        The thread (if `background=True`), otherwise None.
    """
    mv       = cfg["model_version"]
    dest_dir = pathlib.Path(cfg.get("log_dir", ""))
    dest_dir.mkdir(parents=True, exist_ok=True)

    EXIT_TOKEN_LEN = 0  # 4-byte length of zero means DONE

    def _handle_conn(conn: socket.socket):
        with conn:
            while True:
                hdr_len_bytes = conn.recv(4)
                if not hdr_len_bytes:
                    break                         # connection closed early
                hdr_len = struct.unpack(">I", hdr_len_bytes)[0]
                if hdr_len == EXIT_TOKEN_LEN:     # graceful EOF
                    break

                header = json.loads(conn.recv(hdr_len).decode())
                fname, fsize = header["name"], header["size"]

                remaining, chunks = fsize, []
                while remaining:
                    chunk = conn.recv(min(65536, remaining))
                    if not chunk:
                        raise RuntimeError("socket closed mid-file")
                    chunks.append(chunk)
                    remaining -= len(chunk)

                # rename to LOCAL model_version
                if "shared_memory_log_receiver_ppo_" in fname:
                    new_name = f"shared_memory_log_receiver_ppo_{mv}.csv"
                elif "timed_log_write_ppo_" in fname:
                    new_name = f"timed_log_write_ppo_{mv}.csv"
                else:
                    new_name = fname

                (dest_dir / new_name).write_bytes(b"".join(chunks))
                print(f"[listener] saved {new_name}")

    def _server():
        with socket.socket() as s:
            s.bind((host, port))
            s.listen(num_workers)
            print(f"[listener] waiting on {host}:{port}")
            conn, _ = s.accept()
            _handle_conn(conn)
        print("[listener] all logs received – exiting")

    if background:
        th = threading.Thread(target=_server, daemon=True)
        th.start()
        return th
    else:
        _server()
        return None