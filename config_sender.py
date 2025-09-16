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
    "data_dir": "/mnt/nvme0n1/src/",
    "B": 10, # severity of the packet loss punishment
    "K": 1.02, # cost of increasing concurrency
    "loglevel": "info",
    "probing_sec": 3, # probing interval in seconds
    "network_limit": -1, # Network limit (Mbps) per thread
    "io_limit": -1, # I/O limit (Mbps) per thread
    "memory_use": {
        "maximum": 1000,
        "threshold": 1,
    },
    "fixed_probing": {
        "bsize": 10,
        "thread": 3
    },
    "max_cc": {
        "network": 20,
        "io": 20,
        'write': 20
    },
    "mp_opt": True, # use true for ppo, false for gradient
    "method": "ppo", # options: [gradient, ppo]
    "model_version": 'random', # just a tag for the model
    "mode": 'inference', # random or inference
}

mv = configurations["model_version"]
configurations.setdefault("inference_value_model",
                          f"best_models/{mv}_offline_value.pth")
configurations.setdefault("inference_policy_model",
                          f"best_models/{mv}_offline_policy.pth")
configurations.setdefault("max_episodes",
                            20 if configurations["mode"] == "random" else 20000)
configurations.setdefault("multiplier",
                            20 if configurations["mode"] == "random" else 1)            
