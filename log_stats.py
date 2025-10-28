import pandas as pd, pathlib, math
def extract_log_metrics(model_version: str,
                        top_tppt: int = 51,
                        top_tp: int = 101,
                        top_mem: int = 101,
                        base_dir: str = ".") -> dict:
    """
    Scan the three timed throughput logs + the two shared-memory logs that
    belong to *ppo_<model_version>* and return the values needed by
    NetworkSystemSimulator.

    Returns a dict with keys:
      sender_buffer_capacity, receiver_buffer_capacity,
      read_throughput_per_thread, network_throughput_per_thread,
      write_throughput_per_thread, read_bandwidth, network_bandwidth,
      write_bandwidth
    """

    oneGB = 1024
    tag = f"ppo_{model_version}"
    p = pathlib.Path(base_dir)
    cat = "network"

    # ------------ timed_log_* ---------------
    dfs = {}
    f = p / f"timed_log_{cat}_{tag}.csv"
    if f.exists():
        dfs[cat] = pd.read_csv(
            f, header=None,
            names=["current_time", "time_since_beginning",
                    "throughputs", "threads"])
    else:
        raise FileNotFoundError(f"missing {f}")

    tppt, tp = {}, {}
    for cat, df in dfs.items():
        df["tp_per_thread"] = df["throughputs"] / df["threads"]

        nz_tppt = df.loc[df["tp_per_thread"] > 0, "tp_per_thread"]
        k = min(len(nz_tppt), top_tppt)
        tppt[cat] = nz_tppt.nlargest(k).median()

        nz_tp = df.loc[df["throughputs"] > 0, "throughputs"]
        k = min(len(nz_tp), top_tp)
        tp[cat] = nz_tp.nlargest(k).median()

    return {
        "network_throughput_per_thread": int(tppt["network"]),
        "network_bandwidth": int(tp["network"]),
        }
