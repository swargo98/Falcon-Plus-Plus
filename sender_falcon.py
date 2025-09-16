## Only supports Concurrency optimization

import os
import time
import uuid
import socket
import warnings
import datetime
import numpy as np
import psutil
import pprint
import argparse
import logging as log
import multiprocessing as mp
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from config_sender import configurations
from search import  base_optimizer, brute_force, hill_climb, cg_opt, gradient_opt_fast
from network_stats.usage_gemini import collect_metrics
from models import StateSnapshot, snapshot_from_schema1, snapshot_from_schema2, AgentState, Action, SAO, BoundedDelta, Throughput
from reasoner import PolicyLLM, apply_caps, ollama_json_generator
from utils import tcp_stats, get_dir_size
from memory import SAOMemory
import math
warnings.filterwarnings("ignore", category=FutureWarning)
configurations["cpu_count"] = mp.cpu_count()
configurations["thread_limit"] = min(max(1,configurations["max_cc"]['network']), configurations["cpu_count"])

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


def enqueue_chunks(chunk_bytes, file_ids):
    """
    Enqueue virtual chunks as work items: (file_id, start_offset, size).
    Also populates pending_chunks[file_id] so we can detect per-file completion.
    """
    for fid in file_ids:
        total = int(file_sizes[fid])
        nchunks = int(math.ceil(total / float(chunk_bytes)))
        pending_chunks[fid] = nchunks
        start = 0
        while start < total:
            sz = min(chunk_bytes, total - start)
            q.put((fid, start, sz))
            start += sz

def _mark_chunk_done(fid):
    with pending_lock:
        remain = int(pending_chunks.get(fid, 0)) - 1
        pending_chunks[fid] = remain
        if remain <= 0:
            # whole file finished
            file_incomplete.value -= 1

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


def worker(process_id, q):
    while file_incomplete.value > 0:
        if process_status[process_id] == 0:
            time.sleep(0.3)
            pass
        else:
            while num_workers.value < 1:
                time.sleep(0.3)
                pass

            log.debug("Start Process :: {0}".format(process_id))
            sock = None
            try:
                sock = socket.socket()
                sock.settimeout(3)
                sock.connect((HOST, PORT))

                # print("Connected to Receiver: {0}:{1}".format(HOST, PORT))

                if emulab_test:
                    target, factor = 2500, 10
                    max_speed = (target * 1000 * 1000)/8
                    second_target, second_data_count = int(max_speed/factor), 0

                while (not q.empty()) and (process_status[process_id] == 1):
                    try:
                        # print("Queue size: {0}".format(q.qsize()))
                        item = q.get()
                        # print("Got item: {0}".format(item))

                    except:
                        process_status[process_id] = 0
                        break

                    file_id, start_offset, chunk_len = item
                    filename_rel = file_names[file_id]
                    filename_abs = os.path.join(root, filename_rel)
                    total_size = int(file_sizes[file_id])

                    print("Sending file: {0}, offset: {1}, size: {2}".format(
                        filename_rel, start_offset, chunk_len))
                    
                    if chunk_len <= 0:
                    # spurious empty item; treat as done to avoid leaks
                        _mark_chunk_done(file_id)
                        continue

                    if process_status[process_id] == 1:
                        file = open(filename_abs, "rb")
                        header = f"{filename_rel},{int(start_offset)},{int(chunk_len)},{int(total_size)}\n"
                        # print("Header: {0}".format(header))
                        sock.send(header.encode())
                        # print("Sent header")

                        log.debug("sending chunk :: pid=%s file=%s off=%s size=%s",
                          process_id, filename_rel, start_offset, chunk_len)

                        offset = int(start_offset)
                        remaining = int(chunk_len)
                        paused_mid_chunk = False

                        timer100ms = time.time()

                        while (remaining > 0) and (process_status[process_id] == 1):
                            if emulab_test:
                                block_size = min(chunk_size, second_target-second_data_count)
                                data_to_send = bytearray(int(block_size))
                                sent = sock.send(data_to_send)
                            else:
                                block_size = min(chunk_size, remaining)
                                if process_status[process_id] == 0:
                                    # PAUSE requested: requeue remainder of THIS chunk
                                    paused_mid_chunk = True
                                    break
                                if file_transfer:
                                    try:
                                        sent = sock.sendfile(file=file, offset=int(offset), count=int(block_size))
                                        log.debug("Sent data: {0}".format(sent))
                                    except Exception as e:
                                        log.debug("pid=%s send error: %s", process_id, e)
                                        paused_mid_chunk = True
                                        break
                                else:
                                    data_to_send = bytearray(int(block_size))
                                    sent = sock.send(data_to_send)

                            if sent is None:
                                sent = 0
                            
                            offset += sent
                            remaining -= sent
                            file_offsets[file_id] += sent
                            bytes_sent_map[file_id] = int(bytes_sent_map.get(file_id, 0)) + sent

                            if emulab_test:
                                second_data_count += sent
                                if second_data_count >= second_target:
                                    second_data_count = 0
                                    while timer100ms + (1/factor) > time.time():
                                        pass

                                    timer100ms = time.time()

                    if remaining > 0 or paused_mid_chunk:
                        print("Re-enqueueing remaining chunk: {0}, offset: {1}, size: {2}".format(
                            filename_rel, offset, remaining))
                        q.put((file_id, offset, remaining))
                    else:
                        print("Finished sending file: {0}, offset: {1}, size: {2}".format(
                            filename_rel, start_offset, chunk_len))
                        _mark_chunk_done(file_id)

                sock.close()

            except socket.timeout as e:
                pass

            except Exception as e:
                process_status[process_id] = 0
                log.debug("Process: {0}, Error: {1}".format(process_id, str(e)))

            log.debug("End Process :: {0}".format(process_id))

    process_status[process_id] = 0


def sample_transfer(params):
    global throughput_logs, exit_signal

    if file_incomplete.value == 0:
        return exit_signal

    params = [1 if x<1 else int(np.round(x)) for x in params]
    log.info("Sample Transfer -- Probing Parameters: {0}".format(params))
    num_workers.value = params[0]

    current_cc = np.sum(process_status)
    for i in range(configurations["thread_limit"]):
        if i < params[0]:
            if (i >= current_cc):
                process_status[i] = 1
        else:
            process_status[i] = 0

    log.debug("Active CC: {0}".format(np.sum(process_status)))

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

    if configurations["method"].lower() == "llm":
            log.info("Running llm Optimization .... ")
            optimizer = LLMOptimizer()

    elif configurations["method"].lower() == "brute":
        log.info("Running Brute Force Optimization .... ")
        params = brute_force(configurations, sample_transfer, log)

    elif configurations["method"].lower() == "hill_climb":
        log.info("Running Hill Climb Optimization .... ")
        params = hill_climb(configurations, sample_transfer, log)

    elif configurations["method"].lower() == "gradient":
        log.info("Running Gradient Optimization .... ")
        params = gradient_opt_fast(configurations['max_cc']['network'], sample_transfer, log)

    elif configurations["method"].lower() == "cg":
        log.info("Running Conjugate Optimization .... ")
        params = cg_opt(configurations, sample_transfer)

    elif configurations["method"].lower() == "probe":
        log.info("Running a fixed configurations Probing .... ")
        params = [configurations["fixed_probing"]["thread"]]

    else:
        log.info("Running Bayesian Optimization .... ")
        params = gradient_opt_fast(configurations['max_cc']['network'], sample_transfer, log)


    if file_incomplete.value > 0:
        normal_transfer(params)


class LLMOptimizer:
    def __init__(self):
        self.prev_network_throughput = 0
        self.prev_network_thread = 2
        self.prev_reward = 0
        # self.used_disk = get_dir_size(log, tmpfs_dir)
        self.current_network_thread = 2
        self.current_network_throughput = 0
        self.current_reward = 0

        self.utility_network = 0

        self.K = configurations["K"]

        self.policy = PolicyLLM(generate_func=ollama_json_generator("driaforall/tiny-agent-a:0.5b"))
        self.caps = BoundedDelta()  # uses your config caps
        self.memory = SAOMemory(maxlen=5, summary_every=3)

        print("=== LLM OPTIMIZER (OBSERVE→REASON→ACT→EVALUATE) ===")
        self.optimize(log, verbose=True)
    
    def optimize(self, logger, verbose=True):
        print("Starting LLM Optimize Loop")
        st = self.get_state(is_start=True)
        action=Action(concurrency_network=self.current_network_thread)
        while True:
            # ACT (stub)
            print(f"[ACT] Applying -> network={action.concurrency_network}")
            perf_u, Tr, Tn, Tw = self.get_reward([action.concurrency_network, action.concurrency_network, action.concurrency_network])
            if perf_u == exit_signal or Tr == exit_signal or Tn == exit_signal or Tw == exit_signal:
                break
            delta = 0.0 if st.last_reward is None else (perf_u - st.last_reward)
            print(f"[EVALUATE] Tr={Tr:.2f}MB/s Tn={Tn:.2f}MB/s Tw={Tw:.2f}MB/s  "
                f"U={perf_u:.2f}  Δ={delta:.2f}")

            # Log SAO
            st.history.append(
                SAO(
                    sender_state=st.sender_state_snapshot,
                    action=action,
                    throughputs=Throughput(Tn=Tn),
                    utility=perf_u,
                )
            )
            history_window = 3
            if len(st.history) > history_window:
                st.history = st.history[-history_window:]
            self.memory.append(st.history[-1])
            st.last_action = action
            st.last_reward = perf_u

            st = self.get_state(history=st.history)

            # REASON
            print("Inside LLM Optimize Loop")
            t0 = time.time()
            proposal = self.policy.propose(st.sender_state_snapshot, st.last_action, st.last_reward, self.memory)
            t1 = time.time()
            print(f"[REASON] LLM response time: {t1 - t0:.2f}s")
            action = apply_caps(proposal, st.last_action, self.caps)
            print(f"[REASON] proposal={proposal.model_dump()} → capped={action.model_dump()}")

        return -1

    def get_state(self, is_start=False, history=[]):
        network_thread = self.current_network_thread
        read_thread = self.current_network_thread
        write_thread = self.current_network_thread

        network_thrpt = self.current_network_throughput
        read_thrpt = self.current_network_throughput
        write_thrpt = self.current_network_throughput
        # free_disk = (memory_limit - self.used_disk) # NEED TO INTEGRATE LATER w throughputs######
        sender_metrics = collect_metrics(
            sport=None,
            dport=None,
            iface=None,
            exclude_names=["automdt"],
            debug=True
        )
        if is_start:
            receiver_metrics = sender_metrics ###NEEDED TO BE FIXED LATER
        print("Throughputs -- I/O: {0}, Network: {1}, Write: {2}".format(read_thrpt, network_thrpt, write_thrpt))
        print(sender_metrics)    
        
        sender_snapshot = snapshot_from_schema2(sender_metrics)
        receiver_snapshot = snapshot_from_schema2(sender_metrics)

        st = AgentState(
            sender_state_snapshot=sender_snapshot,
            receiver_state_snapshot=receiver_snapshot,
            last_action=Action(concurrency_read=read_thread, concurrency_network=network_thread, concurrency_write=write_thread),
            last_reward=self.current_reward,
            history=history,
        )
        return st
    
    def llm_transfer(self, params):
        print("Inside LLM Transfer")
        global throughput_logs, exit_signal

        if file_incomplete.value == 0:
            print("Exiting LLM Transfer")
            return [exit_signal, None]

        params = [1 if x<1 else int(np.round(x)) for x in params]
        log.info("Sample Transfer -- Probing Parameters: {0}".format(params))
        num_workers.value = params[0]

        current_cc = np.sum(process_status)
        for i in range(configurations["thread_limit"]):
            if i < params[0]:
                if (i >= current_cc):
                    process_status[i] = 1
            else:
                process_status[i] = 0

        log.debug("Active CC: {0}".format(np.sum(process_status)))

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
            return [exit_signal, None]
        else:
            return [score_value, thrpt]
    
    def get_reward(self, params):
        utility, net_thrpt = self.llm_transfer(params)
        io_thrpt, write_thrpt = net_thrpt, net_thrpt
        read_thread, network_thread, write_thread = map(int, params)

        log.info(f"Throughputs -- I/O: {io_thrpt}, Network: {net_thrpt}, Write: {write_thrpt}")

        if io_thrpt == exit_signal or write_thrpt == exit_signal:
            return exit_signal, None, None, None

        self.prev_network_thread = self.current_network_thread
        self.prev_network_throughput = self.current_network_throughput
        self.prev_reward = self.current_reward
        
        
        self.current_network_thread = network_thread
        self.current_network_throughput = net_thrpt
        # self.used_disk = used_disk

        self.current_reward = utility

        return utility, self.current_network_throughput, self.current_network_throughput, self.current_network_throughput
 
def report_throughput(start_time):
    global throughput_logs
    previous_total = 0
    previous_time = 0

    while file_incomplete.value > 0:
        t1 = time.time()
        time_since_begining = np.round(t1-start_time, 1)

        if time_since_begining >= 0.1:
            if time_since_begining >= 30 and sum(throughput_logs[-30:]) == 0:
                file_incomplete.value = 0

            # if time_since_begining >= 60:
            #     file_incomplete.value = 0

            # cpus.append(psutil.cpu_percent())
            # log.info(f"cpu: curr - {np.round(cpus[-1], 4)}, avg - {np.round(np.mean(cpus), 4)}")

            total_bytes = np.sum(file_offsets)
            thrpt = np.round((total_bytes*8)/(time_since_begining*1000*1000), 2)

            curr_total = total_bytes - previous_total
            curr_time_sec = np.round(time_since_begining - previous_time, 3)
            curr_thrpt = np.round((curr_total*8)/(curr_time_sec*1000*1000), 2)
            previous_time, previous_total = time_since_begining, total_bytes
            throughput_logs.append(curr_thrpt)
            m_avg = np.round(np.mean(throughput_logs[-60:]), 2)

            log.info("Throughput @{0}s: Current: {1}Mbps, Average: {2}Mbps, 60Sec_Average: {3}Mbps".format(
                time_since_begining, curr_thrpt, thrpt, m_avg))

            t2 = time.time()
            time.sleep(max(0, 1 - (t2-t1)))


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

    q = manager.Queue()

    pending_chunks = manager.dict()         # file_id -> int
    pending_lock   = manager.Lock()
    bytes_sent_map = manager.dict()
    all_file_ids = range(file_count.value)
    enqueue_chunks(500 * 1024 * 1024, all_file_ids)

    print("Total files to send: {0}, Total size: {1} GB".format(
        file_count.value, np.round(np.sum(file_sizes)/(1024*1024*1024), 3)))

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