configurations = {
    "receiver": {
        "host": "192.168.1.2",
        "port": 50028
    },
    "data_dir": "/mnt/nvme0n1/dest/",
    "max_cc": 1024,
    "K": 1.02,
    "probing_sec": 3, # probing interval in seconds
    "file_transfer": False,
    "loglevel": "info",
    "io_limit": -1, # I/O limit (Mbps) per thread
}