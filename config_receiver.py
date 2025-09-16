configurations = {
    "receiver": {
        "host": "192.168.1.2",
        "port": 50028
    },
    "sender": {
        "host": "192.168.1.1",
        "port": 5003
    },
    "rpc_port":"5002",
    "data_dir": "/mnt/nvme0n1/dest/",
    "max_cc": 30,
    "K": 1.02,
    "probing_sec": 3, # probing interval in seconds
    "file_transfer": True,
    "loglevel": "info",
    "io_limit": -1, # I/O limit (Mbps) per thread
    "memory_use": {
        "maximum": 1000,
        "threshold": 1,
    },
    "method": "ppo", # options: [gradient, ppo]
    "model_version": 'residual',
}