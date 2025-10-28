## Only supports Concurrency optimization

import os
import time
import uuid
import socket
import warnings
import datetime
import numpy as np
# import psutil
import pprint
import argparse
import logging as log
import multiprocessing as mp
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from config_sender import configurations
from search import  base_optimizer, brute_force, hill_climb, cg_opt, gradient_opt_fast
from utils import tcp_stats, get_dir_size
import math
from ppo import PPOAgentContinuous, train_ppo, load_model, NetworkOptimizationEnv
from fpp_simulator import SimulatorState

warnings.filterwarnings("ignore", category=FutureWarning)
configurations["cpu_count"] = mp.cpu_count()
# configurations["thread_limit"] = min(max(1,configurations["max_cc"]['network']), configurations["cpu_count"])
configurations["thread_limit"] = 100

if not os.path.exists('logs'):
        os.makedirs('logs')

log_FORMAT = '%(created)f -- %(levelname)s: %(message)s'
log_file = "logs/" + datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + ".log"

if configurations["loglevel"] == "debug":
    log.basicConfig(
        format=log_FORMAT,
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=log.DEBUG,
        # filename=log_file,
        # filemode="w"
        handlers=[
            log.FileHandler(log_file),
            log.StreamHandler()
        ]
    )

    mp.log_to_stderr(log.DEBUG)
else:
    log.basicConfig(
        format=log_FORMAT,
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=log.INFO,
        # filename=log_file,
        # filemode="w"
        handlers=[
            log.FileHandler(log_file),
            log.StreamHandler()
        ]
    )


emulab_test = False
if "emulab_test" in configurations and configurations["emulab_test"] is not None:
    emulab_test = configurations["emulab_test"]

file_transfer = True
if "file_transfer" in configurations and configurations["file_transfer"] is not None:
    file_transfer = configurations["file_transfer"]

def tcp_stats_cwnd(elapsed, rcvr_addr=None):
    """
    Print cwnd stats (min/max/avg/last) for TCP connections matching rcvr_addr (or global RCVR_ADDR).
    Returns the list of cwnd values found. Does not modify other behavior.
    """
    import os, time

    global RCVR_ADDR
    addr = rcvr_addr or RCVR_ADDR
    start = time.time()
    cwnds = []

    try:
        data = os.popen("ss -ti").read().splitlines()
        for i in range(1, len(data)):
            # If the previous line is a socket line that includes the receiver address
            if addr in data[i - 1]:
                # Next line has TCP stats; normalize tokens (strip commas)
                tokens = [t.strip().rstrip(',') for t in data[i].split() if t.strip()]
                for entry in tokens:
                    if entry.startswith("cwnd:"):
                        # cwnd:10 (integer); be tolerant of parsing issues
                        try:
                            cwnds.append(int(entry.split(":", 1)[1]))
                        except ValueError:
                            # Sometimes non-integer? Ignore gracefully.
                            pass
    except Exception as e:
        print(e)

    if cwnds:
        avg_cwnd = sum(cwnds) / len(cwnds)
        fname = 'cwnd_' + configurations['model_version'] +'.csv'
        with open(fname, 'a') as f:
            f.write(f"{elapsed}, min={min(cwnds)}, max={max(cwnds)}, avg={avg_cwnd:.2f}\n")
    else:
        print(f"TCP cwnd: not found for {addr}")

    end = time.time()
    # Optional: mirror your timing log style if you want
    # log.debug("Time taken to collect cwnd stats: {0}ms".format(np.round((end-start)*1000)))

    return cwnds



def tcp_stats():
    global RCVR_ADDR
    start = time.time()
    sent, retm = 0, 0

    try:
        data = os.popen("ss -ti").read().split("\n")
        for i in range(1,len(data)):
            if RCVR_ADDR in data[i-1]:
                parse_data = data[i].split(" ")
                for entry in parse_data:
                    if "data_segs_out" in entry:
                        sent += int(entry.split(":")[-1])

                    if "bytes_retrans" in entry:
                        pass

                    elif "retrans" in entry:
                        retm += int(entry.split("/")[-1])

    except Exception as e:
        print(e)

    end = time.time()
    log.debug("Time taken to collect tcp stats: {0}ms".format(np.round((end-start)*1000)))
    return sent, retm
import socket, time

def connect_with_retries(host, port, attempts=8, delay=0.15, timeout=3.0):
    for i in range(attempts):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Helpful but optional:
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

        s.settimeout(timeout)           # <-- only for connect
        try:
            s.connect((host, port))
            s.settimeout(None)          # <-- IMPORTANT: block for long sends
            # Alternatively: s.settimeout(60.0) if you prefer a finite write timeout
            return s
        except Exception:
            try: s.close()
            except OSError: pass
            if i < attempts - 1:
                time.sleep(delay)
            else:
                raise

def worker(process_id, q):
    while file_incomplete.value > 0:
        if process_status[process_id] == 0:
            time.sleep(0.05)
            continue

        while num_workers.value < 1:
            time.sleep(0.01)

        log.debug("Start Process :: %d", process_id)

        try:
            while (not q.empty()) and (process_status[process_id] == 1):
                try:
                    file_id = q.get_nowait()
                except Exception:
                    process_status[process_id] = 0
                    log.info("Could not find anything in q :: %d", process_id)
                    break

                offset = file_offsets[file_id]
                to_send = file_sizes[file_id] - offset
                if to_send <= 0:
                    file_incomplete.value -= 1
                    continue

                try:
                    sock = connect_with_retries(HOST, PORT, attempts=8, delay=0.15, timeout=3.0)
                except Exception:
                    print(f"Process Could not connect :: {process_id}")
                    # Re-queue the file so another process can try later
                    q.put(file_id)
                    break

                try:
                    filename = root + file_names[file_id]
                    if file_transfer:
                        f = open(filename, "rb")
                    else:
                        f = None

                    # Send ASCII header line
                    msg = f"{file_names[file_id]},{int(offset)},{int(to_send)}\n"
                    sock.sendall(msg.encode("ascii"))

                    factor = 8
                    if configurations['network_limit'] > 0:
                        target = configurations['network_limit']  # Mbps
                        max_speed_bps = (target * 1024 * 1024) // 8
                        second_target = int(max_speed_bps / factor)
                        second_data_count = 0
                        timer = time.time()

                    remaining = to_send
                    while remaining > 0 and process_status[process_id] == 1:
                        if configurations['network_limit'] > 0:
                            block = min(chunk_size, second_target - second_data_count, remaining)
                        else:
                            block = min(chunk_size, remaining)

                        # inside your worker(), after building `block` and before updating offsets:
                        try:
                            if file_transfer:
                                sent = sock.sendfile(file=f, offset=int(offset), count=int(block))
                            else:
                                sent = sock.send(bytearray(int(block)))
                        except (BlockingIOError, InterruptedError):
                            # transient; try again next iteration
                            continue
                        except socket.timeout:
                            # receiver stalled; requeue and abandon this connection gracefully
                            q.put(file_id)
                            raise
                        except BrokenPipeError:
                            # peer closed; requeue and let outer handler recreate connection
                            q.put(file_id)
                            raise

                        if sent == 0:
                            # treat as stalled/closed; requeue and bail so another process can resume
                            q.put(file_id)
                            raise RuntimeError("socket send returned 0 bytes")


                        offset += sent
                        remaining -= sent
                        file_offsets[file_id] = offset

                        if configurations['network_limit'] > 0:
                            second_data_count += sent
                            if second_data_count >= second_target:
                                second_data_count = 0
                                # simple pacing
                                while timer + (1 / factor) > time.time():
                                    time.sleep(0.001)
                                timer = time.time()

                    if remaining > 0:
                        # Incomplete—let someone else resume
                        q.put(file_id)
                    else:
                        file_incomplete.value -= 1

                finally:
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    try:
                        sock.close()
                    except OSError:
                        pass
                    if file_transfer and f:
                        try:
                            f.close()
                        except Exception:
                            pass

        except Exception as e:
            # process_status[process_id] = 0
            log.error("Process: %d, Error: %s, File_inc: %d", process_id, e, file_incomplete.value)

        log.debug("End Process :: %d", process_id)

    process_status[process_id] = 0

from queue import Empty

def worker_(process_id, q):
    stall_max = 10.0  # seconds without progress before we bail/requeue
    while file_incomplete.value > 0:
        if process_status[process_id] == 0:
            time.sleep(0.05); continue
        while num_workers.value < 1:
            time.sleep(0.01)

        try:
            # Block a bit; don’t rely on .empty()
            file_id = q.get(timeout=0.5)
        except Empty:
            # No work right now; try again without disabling this worker
            continue
        except Exception:
            continue

        offset = file_offsets[file_id]
        remaining = file_sizes[file_id] - offset
        if remaining <= 0:
            file_incomplete.value -= 1
            continue

        # Connect
        try:
            sock = connect_with_retries(HOST, PORT, attempts=8, delay=0.15, timeout=3.0)
            # After connect, use a finite write-timeout to escape stalls
            sock.settimeout(20.0)
        except Exception:
            # Couldn’t connect now; put it back for someone else
            q.put(file_id)
            continue

        f = open(root + file_names[file_id], "rb") if file_transfer else None
        last_advanced = time.time()
        try:
            # header
            hdr = f"{file_names[file_id]},{int(offset)},{int(remaining)}\n"
            sock.sendall(hdr.encode("ascii"))

            while remaining > 0 and process_status[process_id] == 1:
                block = min(chunk_size, remaining)
                try:
                    sent = (sock.sendfile(file=f, offset=int(offset), count=int(block))
                            if file_transfer else sock.send(bytearray(int(block))))
                except (ConnectionResetError, BrokenPipeError, socket.timeout):
                    # transient; requeue work and break
                    q.put(file_id)
                    break

                if sent is None:
                    sent = 0  # some platforms return None for sendfile

                if sent <= 0:
                    # No forward progress: treat as stall
                    if time.time() - last_advanced > stall_max:
                        q.put(file_id)
                        break
                    time.sleep(0.01)
                    continue

                # made progress
                offset += sent
                remaining -= sent
                file_offsets[file_id] = offset
                last_advanced = time.time()

            # finished?
            if remaining <= 0:
                file_incomplete.value -= 1
            elif process_status[process_id] == 1:
                # not done but we bailed (stall/reset): requeue
                q.put(file_id)

        finally:
            try: sock.shutdown(socket.SHUT_RDWR)
            except OSError: pass
            try: sock.close()
            except OSError: pass
            if f: 
                try: f.close()
                except: pass
    process_status[process_id] = 0


def sample_transfer(params):
    global throughput_logs, exit_signal

    if file_incomplete.value == 0:
        return exit_signal

    params = [1 if x<1 else int(np.round(x)) for x in params]
    # params = [15]
    log.info("Sample Transfer -- Probing Parameters: {0}".format(params))
    num_workers.value = params[0]

    current_cc = np.sum(process_status)
    log.info("Active CC Prev: {0}".format(np.sum(process_status)))
    for i in range(configurations["thread_limit"]):
        if i < params[0]:
            process_status[i] = 1
        else:
            process_status[i] = 0

    log.info("Active CC After: {0}".format(np.sum(process_status)))

    time.sleep(1)
    prev_sc, prev_rc = tcp_stats()
    n_time = time.time() + probing_time - 1.1
    # time.sleep(n_time)
    while (time.time() < n_time) and (file_incomplete.value > 0):
        time.sleep(0.1)

    curr_sc, curr_rc = tcp_stats()
    sc, rc = curr_sc - prev_sc, curr_rc - prev_rc

    log.debug("TCP Segments >> Send Count: {0}, Retrans Count: {1}".format(sc, rc))
    seconds_to_consider = max(probing_time - 2, 2)
    thrpt = np.mean(throughput_logs[-seconds_to_consider:]) if len(throughput_logs) > seconds_to_consider else 0

    lr, B, K = 0, int(configurations["B"]), float(configurations["K"])
    if sc != 0:
        lr = rc/sc if sc>rc else 0

    # score = thrpt
    plr_impact = B*lr
    # cc_impact_lin = (K-1) * num_workers.value
    # score = thrpt * (1- plr_impact - cc_impact_lin)
    cc_impact_nl = K**num_workers.value
    score = (thrpt/cc_impact_nl) - (thrpt * plr_impact)
    score_value = np.round(score * (-1))

    log.info("Sample Transfer -- Throughput: {0}Mbps, Loss Rate: {1}%, Score: {2}".format(
        np.round(thrpt), np.round(lr*100, 2), score_value))

    if file_incomplete.value == 0:
        return exit_signal
    else:
        return score_value


def normal_transfer(params):
    num_workers.value = max(1, int(np.round(params[0])))
    log.info("Normal Transfer -- Probing Parameters: {0}".format([num_workers.value]))

    for i in range(num_workers.value):
        process_status[i] = 1

    while (np.sum(process_status) > 0) and (file_incomplete.value > 0):
        pass


def run_transfer():
    params = [2]

    if configurations["method"].lower() == "ppo":
        log.info("Running PPO Optimization .... ")
        optimizer = PPOOptimizer()
    
    elif configurations["method"].lower() == "brute":
        log.info("Running Brute Force Optimization .... ")
        params = brute_force(configurations, sample_transfer, log)

    elif configurations["method"].lower() == "hill_climb":
        log.info("Running Hill Climb Optimization .... ")
        params = hill_climb(configurations, sample_transfer, log)

    elif configurations["method"].lower() == "gradient":
        log.info("Running Gradient Optimization .... ")
        params = gradient_opt_fast(configurations['thread_limit'], sample_transfer, log)

    elif configurations["method"].lower() == "cg":
        log.info("Running Conjugate Optimization .... ")
        params = cg_opt(configurations, sample_transfer)

    elif configurations["method"].lower() == "probe":
        log.info("Running a fixed configurations Probing .... ")
        params = [configurations["fixed_probing"]["thread"]]

    else:
        log.info("Running Bayesian Optimization .... ")
        params = gradient_opt_fast(configurations['thread_limit'], sample_transfer, log)


    if file_incomplete.value > 0:
        normal_transfer(params)

class PPOOptimizer:
    def __init__(self):
        self.prev_network_throughput = 0
        self.prev_network_thread = 2
        self.prev_reward = 0
        self.current_network_thread = 2
        self.current_network_throughput = 0
        self.current_reward = 0

        oneGB = 1024
        self.optimal_network_thread = 5

        self.utility_network = 0

        self.K = configurations["K"]

        self.history_length = 3
        self.obs_dim = 5 + 7 * self.history_length

        state = self.get_state(is_start=True)


        self.env = NetworkOptimizationEnv(black_box_function=self.get_reward, state=state, history_length=self.history_length)
        self.agent = PPOAgentContinuous(state_dim=2, action_dim=1, lr=1e-4, eps_clip=0.1)

        policy_model = ""
        value_model = ""
        is_inference = False
        is_random = False

        if configurations['mode'] == 'inference':
            is_inference = True
            policy_model = configurations['inference_policy_model']
            value_model = configurations['inference_value_model']
        else:
            is_random = True

        if not is_random:
            load_model(self.agent, policy_model, value_model)
        log.info(f"Model loaded successfully. Value: {value_model}, Policy: {policy_model}")

        rewards = train_ppo(self.env, self.agent, max_episodes=configurations['max_episodes'], is_inference = is_inference, is_random = is_random)

        print("Training finished.")

    def get_state(self, is_start=False):
        network_thrpt = self.current_network_throughput
        network_thread = self.current_network_thread


        state = SimulatorState(network_throughput=network_thrpt,
                               network_thread=network_thread
                               )
        return state
    
    def ppo_probing(self, params):
        global throughput_logs, exit_signal

        if file_incomplete.value == 0:
            print("408 File transfer completed.")
            return [exit_signal, exit_signal]

        params = [1 if x<1 else int(np.round(x)) for x in params]
        log.info("412 Probing Parameters: {0}".format(params))
        num_workers.value = params[0]

        current_cc = np.sum(process_status)
        for i in range(configurations["thread_limit"]):
            if i < params[0]:
                    process_status[i] = 1
            else:
                process_status[i] = 0

        log.info("Active CC: {0}".format(np.sum(process_status)))

        time.sleep(1)
        prev_sc, prev_rc = tcp_stats()
        n_time = time.time() + probing_time - 1.1
        # time.sleep(n_time)
        while (time.time() < n_time) and (file_incomplete.value > 0):
            time.sleep(0.1)

        curr_sc, curr_rc = tcp_stats()
        sc, rc = curr_sc - prev_sc, curr_rc - prev_rc

        log.debug("TCP Segments >> Send Count: {0}, Retrans Count: {1}".format(sc, rc))
        thrpt = np.mean(throughput_logs[-2:]) if len(throughput_logs) > 2 else 0

        lr, B, K = 0, int(configurations["B"]), float(configurations["K"])
        if sc != 0:
            lr = rc/sc if sc>rc else 0

        # score = thrpt
        plr_impact = B*lr
        # cc_impact_lin = (K-1) * num_workers.value
        # score = thrpt * (1- plr_impact - cc_impact_lin)
        cc_impact_nl = K**num_workers.value
        score = (thrpt/cc_impact_nl) - (thrpt * plr_impact)
        score_value = np.round(score * (-1))

        log.info("Sample Transfer -- Throughput: {0}Mbps, Loss Rate: {1}%, Score: {2}".format(
            np.round(thrpt), np.round(lr*100, 2), score_value))

        if file_incomplete.value == 0:
            print("454 File transfer completed.")
            return [exit_signal, exit_signal]
        else:
            return [thrpt, score_value]

    def get_reward(self, params):
        net_thrpt, reward = self.ppo_probing(params)
        network_thread = params[0]

        log.info(f"Throughputs -- Network: {net_thrpt}")

        if net_thrpt == exit_signal:
            return exit_signal, None

        self.prev_network_thread = self.current_network_thread
        self.prev_network_throughput = self.current_network_throughput
        self.prev_reward = self.current_reward
        self.current_network_thread = network_thread
        self.current_network_throughput = net_thrpt


        # utility_network = (net_thrpt/self.K ** network_thread)

        # reward = utility_network
        self.current_reward = reward


        # network_grad = (utility_network-self.utility_network)/(network_thread-self.prev_network_thread) if (network_thread-self.prev_network_thread) > 0 else 0
        # grads = [network_grad]
        # grads = np.array(grads, dtype=np.float32)

        # self.utility_network = utility_network

        final_state = self.get_state()

        return reward, final_state
        


def report_throughput(start_time):
    global throughput_logs
    previous_total = 0
    previous_time = 0
    no_progress_secs = 0

    while file_incomplete.value > 0:
        t1 = time.time()
        elapsed = np.round(t1 - start_time, 1)

        total_bytes = np.sum(file_offsets)
        curr_total = total_bytes - previous_total
        curr_time = max(1e-3, elapsed - previous_time)
        curr_thrpt = np.round((curr_total*8)/(curr_time*1_000_000), 2)

        previous_total, previous_time = total_bytes, elapsed
        throughput_logs.append(curr_thrpt)
        m_avg = np.round(np.mean(throughput_logs[-60:]), 2)

        if curr_thrpt <= 0.01:
            no_progress_secs += 1
        else:
            no_progress_secs = 0

        # WARN only; do NOT flip file_incomplete
        if no_progress_secs == 60:   # or 120 if you prefer
            log.warning("No progress for 60s (queue may be stuck, receiver backlogged, or tuning overshot). "
                        "Reducing num_workers to 1 temporarily.")
            num_workers.value = 1   # gentle nudge, not a kill

        log.info("Throughput @%ss: Current: %sMbps, Average: %sMbps, 60Sec_Average: %sMbps",
                 elapsed, curr_thrpt, np.round((total_bytes*8)/(elapsed*1000000), 2), m_avg)
        

        t2 = time.time()
        tcp_stats_cwnd(elapsed)
        fname = 'timed_log_network_ppo_' + configurations['model_version'] +'.csv'
        with open(fname, 'a') as f:
            f.write(f"{t2}, {elapsed}, {curr_thrpt}, {sum(process_status)}\n")
        time.sleep(max(0, 1 - (t2 - t1)))


if __name__ == '__main__':
    pp = pprint.PrettyPrinter(indent=4)
    parser=argparse.ArgumentParser()
    parser.add_argument("--host", help="Receiver Host Address")
    parser.add_argument("--port", help="Receiver Port Number")
    parser.add_argument("--data_dir", help="Sender Data Directory")
    parser.add_argument("--method", help="choose one of them : gradient, bayes, brute, probe")
    args = vars(parser.parse_args())
    # pp.pprint(f"Command line arguments: {args}")

    if args["host"]:
        configurations["receiver"]["host"] = args["host"]

    if args["port"]:
        configurations["receiver"]["port"] = int(args["port"])

    if args["data_dir"]:
        configurations["data_dir"] = args["data_dir"]

    if args["method"]:
        configurations["method"] = args["method"]

    pp.pprint(configurations)

    manager = mp.Manager()
    root = configurations["data_dir"]
    probing_time = configurations["probing_sec"]
    file_names = os.listdir(root) * configurations["multiplier"]
    file_sizes = [os.path.getsize(root+filename) for filename in file_names]
    file_count = mp.Value("i",len(file_names))
    throughput_logs = manager.list()

    exit_signal = 10 ** 10
    chunk_size = 1 * 1024 * 1024
    num_workers = mp.Value("i", 0)
    file_incomplete = mp.Value("i", file_count.value)
    process_status = mp.Array("i", [0 for i in range(configurations["thread_limit"])])
    file_offsets = mp.Array("d", [0.0 for i in range(file_count.value)])
    cpus = manager.list()

    HOST, PORT = configurations["receiver"]["host"], configurations["receiver"]["port"]
    RCVR_ADDR = str(HOST) + ":" + str(PORT)

    q = manager.Queue(maxsize=file_count.value)
    for i in range(file_count.value):
        q.put(i)

    workers = [mp.Process(target=worker, args=(i, q)) for i in range(configurations["thread_limit"])]
    for p in workers:
        p.daemon = True
        p.start()

    start = time.time()
    reporting_process = mp.Process(target=report_throughput, args=(start,))
    reporting_process.daemon = True
    reporting_process.start()
    run_transfer()
    end = time.time()

    time_since_begining = np.round(end-start, 3)
    total = np.round(np.sum(file_offsets) / (1024*1024*1024), 3)
    thrpt = np.round((total*8*1024)/time_since_begining,2)
    log.info("Total: {0} GB, Time: {1} sec, Throughput: {2} Mbps".format(
        total, time_since_begining, thrpt))

    reporting_process.terminate()
    for p in workers:
        if p.is_alive():
            p.terminate()
            p.join(timeout=0.1)