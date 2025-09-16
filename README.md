# AutoMDT — Modular Architecture for High‑Performance Data Transfers

AutoMDT is an **agent‑driven** file‑transfer framework that couples a
three‑headed Proximal Policy Optimization (PPO) agent with a lightweight
I/O–network **simulator** to learn the optimal concurrency for **read, network
and write** operations *offline*, then apply the model *online* for production
transfers.

This repository accompanies our INDIS ’25 submission:

> **Modular Architecture for High-Performance and
Low Overhead Data Transfers**  
> Rasman M. Swargo *et al.*, INDIS @ SC 25

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16757552.svg)](https://doi.org/10.5281/zenodo.16757552)

---

## Quick start (two‑machine setup)

| Phase | Receiver (Node B) | Sender (Node A) |
|-------|-------------------|-----------------|
| **Common prep** | <ul><li>`git clone https://github.com/swargo98/AutoMDT-Modular-Architecture-for-High-Performance-Data-Transfers.git`</li><li>`cd AutoMDT-Modular-Architecture-for-High-Performance-Data-Transfers`</li><li>`chmod +x setup.sh && ./setup.sh`</li><li>`source venv/bin/activate`</li><li>`pip install -r requirements.txt`</li><li>Edit **`config_receiver.py`** (IP / port etc.).</li></ul> | <ul><li>Same steps 1–5 as receiver.</li><li>Edit **`config_sender.py`** (IP / port etc.).</li></ul> |
| **Exploration run** | `python receiver.py` | <ul> <li><strong>Generate dataset by modifying <code>file_gen.sh</code></strong> (if needed): <code>./file_gen.sh</code></li> <li>Set <code>"mode": "random"</code> and a new <code>"model_version"</code> in <code>config_sender.py</code>.</li><li><code>python sender.py</code></li></ul> |
| **Offline training** | *(no action)* | `python offline.py` &nbsp;→ trains PPO (≈45 min) |
| **Production / inference** | `python receiver.py` (keep running) | 1. Set `"mode": "inference"` in `config_sender.py`.<br>2. `python sender.py` |

> **Tip :** `receiver.py` must be running **before any transfer** phase.  
> All tunables (ports, buffer sizes, max threads, etc.) live in
> `config_receiver.py` and `config_sender.py`.

---

## Repository layout

```
AutoMDT-Modular-Architecture-for-High-Performance-Data-Transfers/
├── config_receiver.py # Receiver‑side settings
├── config_sender.py # Sender‑side settings
├── file_gen.sh # Utility: create random datasets
├── get_pip.py # Bootstrap pip on bare systems
├── log_stats.py # Parse throughput logs for offline training
├── offline.py # Offline PPO training pipeline
├── ppo.py # PPO agent
├── receiver.py # Receiver daemon
├── receiver_helper.py # Helper routines for receiver
├── requirements.txt # Python dependencies
├── search.py # optimizer
├── sender.py # Sender driver (random / inference)
├── sender_helper.py # Helper routines for sender
├── setup.sh # Creates venv & installs deps
├── utils.py # Shared utilities
└── LICENSE
```

---

## Dependencies

* Linux kernel ≥ 5.x  
* Python 3.9 – 3.12  
* `torch`, `numpy`, `gymnasium`, `tqdm`, `pandas`, `matplotlib`  
* (Optional) `iperf3` for link‑capacity sanity checks

All versions are pinned in **`requirements.txt`**.

---

## Reproducing paper results

1. Use **CloudLab** or **Fabric** nodes with ≥ 10 Gb s⁻¹ NICs.  
2. Follow the *Quick start* table above to train (≈45 min) and then
   run a transfer.  
3. Parse the logs (will be saved in the same directory) to replicate Figure 4 of the paper.

---

## Citing

This section will be updated soon.
<!-- If you build on AutoMDT, please cite:

```bibtex
@inproceedings{Swargo2025AutoMDT,
  title     = {A Modular DRL Architecture for High-Performance and Low-Overhead Data Transfers},
  author    = {Rasman Mubtasim Swargo and Md Arifuzzaman and Engin Arslan},
  booktitle = {Proc. INDIS (SC Workshop)},
  year      = {2025}
}

``` -->

---

## License

MIT — see `LICENSE` for details.
