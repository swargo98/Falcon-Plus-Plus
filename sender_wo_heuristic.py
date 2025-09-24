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
            pass
        else:
            while num_workers.value < 1:
                pass

            log.debug("Start Process :: {0}".format(process_id))
            try:
                sock = socket.socket()
                sock.settimeout(3)
                sock.connect((HOST, PORT))

                if emulab_test:
                    target, factor = 2500, 10
                    max_speed = (target * 1000 * 1000)/8
                    second_target, second_data_count = int(max_speed/factor), 0

                while (not q.empty()) and (process_status[process_id] == 1):
                    try:
                        file_id = q.get()

                    except:
                        process_status[process_id] = 0
                        break

                    offset = file_offsets[file_id]
                    to_send = file_sizes[file_id] - offset

                    if (to_send > 0) and (process_status[process_id] == 1):
                        filename = root + file_names[file_id]
                        file = open(filename, "rb")
                        msg = file_names[file_id] + "," + str(int(offset))
                        msg += "," + str(int(to_send)) + "\n"
                        sock.send(msg.encode())

                        log.debug("starting {0}, {1}, {2}".format(process_id, file_id, filename))
                        timer100ms = time.time()

                        while (to_send > 0) and (process_status[process_id] == 1):
                            if emulab_test:
                                block_size = min(chunk_size, second_target-second_data_count)
                                data_to_send = bytearray(int(block_size))
                                sent = sock.send(data_to_send)
                            else:
                                block_size = min(chunk_size, to_send)
                                if file_transfer:
                                    sent = sock.sendfile(file=file, offset=int(offset), count=int(block_size))
                                    # data = os.preadv(file, block_size, offset)
                                else:
                                    data_to_send = bytearray(int(block_size))
                                    sent = sock.send(data_to_send)

                            offset += sent
                            to_send -= sent
                            file_offsets[file_id] = offset

                            if emulab_test:
                                second_data_count += sent
                                if second_data_count >= second_target:
                                    second_data_count = 0
                                    while timer100ms + (1/factor) > time.time():
                                        pass

                                    timer100ms = time.time()

                    if to_send > 0:
                        q.put(file_id)
                    else:
                        file_incomplete.value = file_incomplete.value - 1

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
        for i in range(configurations["max_cc"]['network']):
            if i < params[0]:
                if (i >= current_cc):
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
            fname = 'timed_log_network_ppo_' + configurations['model_version'] +'.csv'
            with open(fname, 'a') as f:
                f.write(f"{t2}, {time_since_begining}, {curr_thrpt}, {sum(process_status)}\n")
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
    process_status = mp.Array("i", [0 for i in range(configurations["max_cc"]['network'])])
    file_offsets = mp.Array("d", [0.0 for i in range(file_count.value)])
    cpus = manager.list()

    HOST, PORT = configurations["receiver"]["host"], configurations["receiver"]["port"]
    RCVR_ADDR = str(HOST) + ":" + str(PORT)

    q = manager.Queue(maxsize=file_count)
    for i in range(file_count):
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