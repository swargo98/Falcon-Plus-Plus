import os
import shutil
import signal
import time
import socket
import warnings
import datetime
import numpy as np
import logging as logger
import multiprocessing as mp
from threading import Thread
from config_sender import configurations
from search import base_optimizer, hill_climb, cg_opt, gradient_opt_fast, gradient_multivariate
from utils import tcp_stats, run, available_space, get_dir_size
from ppo import SimulatorState, NetworkOptimizationEnv, PPOAgentContinuous, load_model, train_ppo
import contextlib

warnings.filterwarnings("ignore", category=FutureWarning)

def send_file_count(host: str, port: int, total: int) -> None:
    """One-shot handshake that tells the receiver how many source files exist."""
    with contextlib.closing(socket.socket()) as s:
        s.connect((host, port))
        s.sendall(f"COUNT,{total}\n".encode())

def copy_file(process_id):
    global rQueue, tQueue, io_process_status, io_file_offsets, file_names, file_sizes, memory_limit, io_limit, chunk_size, file_transfer, file_processed, file_copied
    while rQueue:
        if io_process_status[process_id] == 1:
            logger.debug(f'Starting Copying Thread: {process_id}')
            try:
                used = get_dir_size(logger, tmpfs_dir)
                if used < memory_limit:
                    file_id, offset = rQueue.popitem() if rQueue else (None, None)
                    if file_id is None:
                        time.sleep(0.1)
                        continue
                    if file_transfer:
                        fname = file_names[file_id]
                        fd = os.open(tmpfs_dir+fname, os.O_CREAT | os.O_RDWR)
                        block_size = chunk_size
                        if io_limit > 0:
                            target, factor = io_limit, 8
                            max_speed = (target * 1024 * 1024)/8
                            second_target, second_data_count = int(max_speed/factor), 0
                            block_size = min(block_size, second_target)
                            timer100ms = time.time()

                        with open(root_dir+fname.split('-')[0], "rb") as ff:
                            ff.seek(int(offset))
                            chunk = ff.read(block_size)

                            os.lseek(fd, int(offset), os.SEEK_SET)
                            while chunk and io_process_status[process_id] == 1:
                                os.write(fd, chunk)
                                offset += len(chunk)

                                # Update every 100 milliseconds
                                io_file_offsets[file_id] = offset

                                if io_limit > 0:
                                    second_data_count += len(chunk)
                                    if second_data_count >= second_target:
                                        second_data_count = 0
                                        while timer100ms + (1/factor) > time.time():
                                            pass

                                        timer100ms = time.time()

                                chunk = ff.read(block_size)

                            io_file_offsets[file_id] = offset
                            if offset < file_sizes[file_id]:
                                logger.debug(f"I/O - file: {file_id}, offset: {offset}, size: {file_sizes[file_id]}")
                                rQueue[file_id] = offset
                            else:
                                logger.debug(f'I/O :: {file_id}')

                            if offset >= file_sizes[file_id] and file_id not in tQueue:
                                file_copied.value += 1
                                tQueue[file_id] = 0

                        os.close(fd)
                    else:
                        io_file_offsets[file_id] = file_sizes[file_id]
                        tQueue[file_id] = 0
            except KeyError:
                time.sleep(0.1)

            except Exception as e:
                logger.debug(str(e))
                time.sleep(0.1)

            logger.debug(f'Exiting Copying Thread: {process_id}')

    io_process_status[process_id] = 0


def transfer_file(process_id):
    global file_processed
    while file_processed.value < file_count:
        if transfer_process_status[process_id] == 1:
            try:
                logger.debug(f'Starting TCP Socket Thread: {process_id}')
                sock = socket.socket()
                sock.settimeout(3)
                sock.connect((HOST, PORT))
            except socket.timeout as e:
                logger.info(f"Socket Timeout: {e}")
                # logger.exception(e)
                continue

            while tQueue and transfer_process_status[process_id] == 1:
                try:
                    file_id, offset = tQueue.popitem() if tQueue else (None, None)
                    if file_id is None:
                        time.sleep(0.1)
                        continue

                    offset = transfer_file_offsets[file_id]

                    if network_limit>0:
                        target, factor = network_limit, 8
                        max_speed = (target * 1024 * 1024)/8
                        second_target, second_data_count = int(max_speed/factor), 0

                    to_send = io_file_offsets[file_id] - offset
                    if (int(to_send) > 0) and (transfer_process_status[process_id] == 1):
                        filename = tmpfs_dir + file_names[file_id]

                        if file_transfer:
                            try:
                                with open(filename, 'rb') as file:
                                    msg = f"{len(rQueue)},{len(tQueue)},{file_names[file_id]},{int(offset)},{int(to_send)}\n"
                                    sock.send(msg.encode())
                                    logger.debug(f"starting {process_id}, {filename}, {offset}, {len(tQueue)}")

                                    timer100ms = offset_update = time.time()
                                    while (int(to_send) > 0) and (transfer_process_status[process_id] == 1):

                                        if network_limit>0:
                                            block_size = min(chunk_size, second_target-second_data_count, to_send)
                                        else:
                                            block_size = min(chunk_size, to_send)

                                        if file_transfer:
                                            sent = sock.sendfile(file=file, offset=int(offset), count=int(block_size))
                                        else:
                                            data_to_send = bytearray(int(block_size))
                                            sent = sock.send(data_to_send)
                                        offset += sent
                                        to_send -= sent

                                        ## Update every 100 milliseconds
                                        if time.time() - offset_update >= 0.1:
                                            transfer_file_offsets[file_id] = offset
                                            offset_update = time.time()

                                        if network_limit>0:
                                            second_data_count += sent
                                            if second_data_count >= second_target:
                                                second_data_count = 0
                                                while timer100ms + (1/factor) > time.time():
                                                    pass

                                                timer100ms = time.time()
                            except Exception as e:
                                logger.info("An error occurred:", e)

                        transfer_file_offsets[file_id] = offset
                            
                        if float(offset) < float(io_file_offsets[file_id]) or file_id in rQueue:
                            logger.debug(f"Transfer - file: {file_id}, offset: {offset}, size: {file_sizes[file_id]}")
                            tQueue[file_id] = transfer_file_offsets[file_id]
                        else:
                            logger.debug(f'Transfer :: {file_id}!')
                            file_processed.value += 1
                            logger.info(f"File Processed: {file_processed.value}")
                            if file_transfer:
                                run(f'rm {filename}', logger)
                                logger.debug(f'Cleanup :: {file_id}!')
                    else:
                        tQueue[file_id] = transfer_file_offsets[file_id]
                except KeyError:
                    time.sleep(0.1)
                    continue

                except Exception as e:
                    transfer_process_status[process_id] = 0
                    logger.error("Process: {0}, Error: {1}".format(process_id, str(e)))

            logger.debug(f'Exiting TCP Socket Thread: {process_id}')
            sock.close()

    transfer_process_status[process_id] = 0


def network_probing(params):
    global network_throughput_logs, exit_signal, file_processed

    if not rQueue and not tQueue:
        return exit_signal

    while not tQueue:
        time.sleep(0.1)

    params = [1 if x<1 else int(np.round(x)) for x in params]
    logger.info("Network -- Probing Parameters: {0}".format(params))
    for i in range(len(transfer_process_status)):
        transfer_process_status[i] = 1 if i < params[0] else 0

    logger.debug("Active CC - Socket: {0}".format(np.sum(transfer_process_status)))
    time.sleep(1)
    prev_sc, prev_rc = tcp_stats(RCVR_ADDR, logger)
    n_time = time.time() + probing_time - 1.05
    while (time.time() < n_time) and (file_processed.value < file_count):
        time.sleep(0.1)

    curr_sc, curr_rc = tcp_stats(RCVR_ADDR, logger)
    sc, rc = curr_sc - prev_sc, curr_rc - prev_rc

    logger.debug("TCP Segments >> Send Count: {0}, Retrans Count: {1}".format(sc, rc))
    thrpt = np.mean(network_throughput_logs[-2:]) if len(network_throughput_logs) > 2 else 0

    lr, B, K = 0, int(configurations["B"]), float(configurations["K"])
    if sc != 0:
        lr = rc/sc if sc>rc else 0

    # score = thrpt
    plr_impact = B*lr
    cc_impact_nl = K**params[0]
    score = (thrpt/cc_impact_nl) - (thrpt * plr_impact)
    score_value = np.round(score * (-1))

    logger.info(f"rQueue:{len(rQueue)}, tQueue:{len(tQueue)}")
    logger.info("Network Probing -- Throughput: {0}Mbps, Loss Rate: {1}%, Score: {2}".format(
        np.round(thrpt), np.round(lr*100, 2), score_value))

    if not rQueue and not tQueue:
        return exit_signal
    else:
        return score_value


def io_probing(params):
    global io_throughput_logs, exit_signal

    if not rQueue:
        return exit_signal

    params = [1 if x<1 else int(np.round(x)) for x in params]
    logger.info("I/O -- Probing Parameters: {0}".format(params))
    for i in range(len(io_process_status)):
        io_process_status[i] = 1 if i < params[0] else 0

    logger.debug("Active CC - I/O: {0}".format(np.sum(io_process_status)))
    time.sleep(1)
    n_time = time.time() + probing_time - 1.05
    used_before = get_dir_size(logger, tmpfs_dir)
    # time.sleep(n_time)
    while (time.time() < n_time) and (file_processed.value < file_count):
        time.sleep(0.1)

    used_disk = get_dir_size(logger, tmpfs_dir)
    logger.info(f"Shared Memory -- Used: {used_disk}GB")
    thrpt = np.mean(io_throughput_logs[-2:]) if len(io_throughput_logs) > 2 else 0
    K = float(configurations["K"])
    limit = min(configurations["memory_use"]["threshold"], memory_limit//2)

    storage_cost = 0
    if used_disk>limit and used_disk > used_before:
        storage_cost = (used_disk - used_before) / used_disk

    cc_impact_nl = K**params[0]
    score = thrpt/cc_impact_nl - thrpt*storage_cost
    score_value = np.round(score * (-1))

    logger.info(f"I/O Probing -- Throughput: {np.round(thrpt)}Mbps, Score: {score_value}")
    if not rQueue:
        return exit_signal
    else:
        return score_value

import zmq

from tinyrpc.protocols.jsonrpc import JSONRPCProtocol
from tinyrpc.transports.zmq import ZmqClientTransport
from tinyrpc import RPCClient

def set_write_thread(write_thread):
    ctx = zmq.Context()
    HOST = configurations["receiver"]["host"]
    port = configurations['rpc_port']
    rpc_client = RPCClient(
        JSONRPCProtocol(),
        ZmqClientTransport.create(ctx, 'tcp://'+HOST+':'+port)
    )

    remote_server = rpc_client.get_proxy()
    remote_server.set_thread(write_thread)

def get_write_state():
    ctx = zmq.Context()
    HOST = configurations["receiver"]["host"]
    port = configurations['rpc_port']
    rpc_client = RPCClient(
        JSONRPCProtocol(),
        ZmqClientTransport.create(ctx, 'tcp://'+HOST+':'+port)
    )

    remote_server = rpc_client.get_proxy()
    result = remote_server.get_state()
    return result

def get_write_throughput():
    ctx = zmq.Context()
    HOST = configurations["receiver"]["host"]
    port = configurations['rpc_port']
    rpc_client = RPCClient(
        JSONRPCProtocol(),
        ZmqClientTransport.create(ctx, 'tcp://'+HOST+':'+port)
    )

    remote_server = rpc_client.get_proxy()
    result = remote_server.get_throughput()
    return result

def exit_write_process():
    ctx = zmq.Context()
    HOST = configurations["receiver"]["host"]
    port = configurations['rpc_port']
    rpc_client = RPCClient(
        JSONRPCProtocol(),
        ZmqClientTransport.create(ctx, 'tcp://'+HOST+':'+port)
    )

    remote_server = rpc_client.get_proxy()
    logger.info("Exiting Server")
    remote_server.exit()
    logger.info("Server answered: Exited")

class PPOOptimizer:
    def __init__(self):
        self.prev_read_throughput = 0
        self.prev_network_throughput = 0
        self.prev_read_thread = 2
        self.prev_network_thread = 2
        self.prev_write_thread = 2
        self.prev_reward = 0
        self.used_disk = get_dir_size(logger, tmpfs_dir)
        self.current_read_thread = 2
        self.current_network_thread = 2
        self.current_write_thread = 2
        self.current_read_throughput = 0
        self.current_network_throughput = 0
        self.current_reward = 0

        oneGB = 1024
        self.optimal_read_thread = 5
        self.optimal_network_thread = 5
        self.optimal_write_thread = 5
        self.stable_bw = 5 * configurations["network_limit"]

        self.utility_read = 0
        self.utility_network = 0
        self.utility_write = 0

        self.K = configurations["K"]

        self.history_length = 3
        self.obs_dim = 5 + 7 * self.history_length

        state = self.get_state(is_start=True)


        self.env = NetworkOptimizationEnv(black_box_function=self.get_reward, state=state, history_length=self.history_length)
        self.agent = PPOAgentContinuous(state_dim=8, action_dim=3, lr=1e-4, eps_clip=0.1)

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
        logger.info(f"Model loaded successfully. Value: {value_model}, Policy: {policy_model}")

        rewards = train_ppo(self.env, self.agent, max_episodes=configurations['max_episodes'], is_inference = is_inference, is_random = is_random)

    def get_state(self, is_start=False):
        write_thrpt, write_free, write_thread = 0, 1, 2
        if not is_start:
            write_thrpt, _, write_free, write_thread = get_write_state()
        read_thrpt = self.current_read_throughput
        network_thrpt = self.current_network_throughput
        read_thread = self.current_read_thread
        network_thread = self.current_network_thread
        free_disk = (memory_limit - self.used_disk)


        state = SimulatorState(sender_buffer_remaining_capacity=free_disk * 1024,
                               receiver_buffer_remaining_capacity=write_free * 1024,
                               read_throughput=read_thrpt,
                               network_throughput=network_thrpt,
                               write_throughput=write_thrpt,
                               read_thread=read_thread,
                               write_thread=write_thread,
                               network_thread=network_thread
                               )
        return state

    def ppo_probing(self, params):
        global io_throughput_logs, network_throughput_logs, exit_signal, rQueue, tQueue, file_processed, file_count

        if file_processed.value == file_count:
            logger.info("Exiting Write 464")
            return [exit_signal, None, None]
        
        read_thread, network_thread, write_thread = map(int, params)

        write_thread_set = Thread(target=set_write_thread, args=(write_thread,), daemon=False)
        write_thread_set.start()

        logger.info("Probing Parameters - [Read, Network, Write]: {0}, {1}, {2}".format(read_thread, network_thread, write_thread))
        
        for i in range(len(transfer_process_status)):
            transfer_process_status[i] = 1 if (i < network_thread and file_processed.value<file_count) else 0

        if params[1]:
            for i in range(len(io_process_status)):
                io_process_status[i] = 1 if (i < read_thread and file_copied.value<file_count) else 0

        time.sleep(1)

        # Before
        prev_sc, prev_rc = tcp_stats(RCVR_ADDR, logger)
        n_time = time.time() + probing_time - 1.05
        while (time.time() < n_time) and (file_processed.value < file_count):
            time.sleep(0.1)

        # After
        curr_sc, curr_rc = tcp_stats(RCVR_ADDR, logger)
        sc, rc = curr_sc - prev_sc, curr_rc - prev_rc
        logger.debug("TCP Segments >> Send Count: {0}, Retrans Count: {1}".format(sc, rc))
        used_disk = get_dir_size(logger, tmpfs_dir)

        write_thread_set.join()

        ## Network Score
        net_thrpt = np.round(np.mean(network_throughput_logs[-2:])) if len(network_throughput_logs) > 2 else 0
        lr, B, K = 0, int(configurations["B"]), float(configurations["K"])

        ## I/O score
        io_thrpt = 0
        if read_thread and file_copied.value<file_count:
            io_thrpt = np.round(np.mean(io_throughput_logs[-2:])) if len(io_throughput_logs) > 2 else 0
        else:
            io_thrpt = 0  

        logger.info(f"Shared Memory -- Used: {used_disk}GB")
        logger.info(f"rQueue:{len(rQueue)}, tQueue:{len(tQueue)}")

        with open('shared_memory_log_sender_ppo_' + configurations['model_version'] +'.csv', 'a') as f:
                f.write(f"{used_disk}\n")

        if file_processed.value == file_count:
            return [exit_signal, None, None]

        logger.info(f"Probing -- I/O: {io_thrpt}Mbps, Network: {net_thrpt}Mbps")
        return [io_thrpt, net_thrpt, used_disk] #score_value

    def get_reward(self, params):
        io_thrpt, net_thrpt, used_disk = self.ppo_probing(params)
        read_thread, network_thread, write_thread = map(int, params)
        write_thrpt = get_write_throughput()

        logger.info(f"Throughputs -- I/O: {io_thrpt}, Network: {net_thrpt}, Write: {write_thrpt}")

        if io_thrpt == exit_signal or write_thrpt == exit_signal:
            logger.info("Exiting Write Process 521")
            exit_write_process()
            logger.info("Exited Write Process 523")
            return exit_signal, None

        self.prev_read_thread = self.current_read_thread
        self.prev_network_thread = self.current_network_thread
        self.prev_write_thread = self.current_write_thread
        self.prev_read_throughput = self.current_read_throughput
        self.prev_network_throughput = self.current_network_throughput
        self.prev_reward = self.current_reward
        
        
        self.current_read_thread = read_thread
        self.current_network_thread = network_thread
        self.current_write_thread = write_thread
        self.current_read_throughput = io_thrpt
        self.current_network_throughput = net_thrpt
        self.used_disk = used_disk


        utility_read = (io_thrpt/self.K ** read_thread)
        utility_network = (net_thrpt/self.K ** network_thread)
        utility_write = (write_thrpt/self.K ** write_thread)

        reward = utility_read + utility_network + utility_write
        self.current_reward = reward


        read_grad = (utility_read-self.utility_read)/(read_thread-self.prev_read_thread) if (read_thread-self.prev_read_thread) > 0 else 0
        network_grad = (utility_network-self.utility_network)/(network_thread-self.prev_network_thread) if (network_thread-self.prev_network_thread) > 0 else 0
        write_grad = (utility_write-self.utility_write)/(write_thread-self.prev_write_thread) if (write_thread-self.prev_write_thread) > 0 else 0
        grads = [read_grad, network_grad, write_grad]
        grads = np.array(grads, dtype=np.float32)

        self.utility_read = utility_read
        self.utility_network = utility_network
        self.utility_write = utility_write

        final_state = self.get_state()

        return reward, final_state
        



def multi_params_probing(params):
    global io_throughput_logs, network_throughput_logs, exit_signal, rQueue, tQueue, file_processed, file_count
    # global io_weight, net_weight

    if not rQueue and not tQueue:
        return [exit_signal, exit_signal]

    params[0] = max(1,  int(np.round(params[0])))
    logger.info("Probing Parameters - [Network, I/O]: {0}".format(params))

    for i in range(len(transfer_process_status)):
        transfer_process_status[i] = 1 if i < params[0] else 0

    if params[1]:
        for i in range(len(io_process_status)):
            io_process_status[i] = 1 if (i < params[1] and rQueue) else 0

    time.sleep(1)

    # Before
    prev_sc, prev_rc = tcp_stats(RCVR_ADDR, logger)
    n_time = time.time() + probing_time - 1.05
    while (time.time() < n_time) and (file_processed.value < file_count):
        time.sleep(0.1)

    # After
    curr_sc, curr_rc = tcp_stats(RCVR_ADDR, logger)
    sc, rc = curr_sc - prev_sc, curr_rc - prev_rc
    logger.debug("TCP Segments >> Send Count: {0}, Retrans Count: {1}".format(sc, rc))
    used_disk = get_dir_size(logger, tmpfs_dir)

    ## Network Score
    net_thrpt = np.round(np.mean(network_throughput_logs[-2:])) if len(network_throughput_logs) > 2 else 0
    lr, B, K = 0, int(configurations["B"]), float(configurations["K"])
    if sc != 0:
        lr = rc/sc if sc>rc else 0

    plr_impact = B*lr
    cc_impact_nl = K**params[0]
    net_score = (net_thrpt/cc_impact_nl) - (net_thrpt * plr_impact)
    net_score_value = np.round(net_score * (-1))

    ## I/O score
    io_thrpt = 0
    if params[1] and rQueue:
        io_thrpt = np.round(np.mean(io_throughput_logs[-2:])) if len(io_throughput_logs) > 2 else 0
        cc_impact_nl = K**params[1]
        io_score = io_thrpt/cc_impact_nl
        io_score_value = np.round(io_score * (-1))
    else:
        io_score_value = exit_signal

    logger.info(f"Shared Memory -- Used: {used_disk}GB")
    logger.info(f"rQueue:{len(rQueue)}, tQueue:{len(tQueue)}")

    if not rQueue and not tQueue:
        net_score_value = exit_signal

    logger.info(f"Probing -- I/O: {io_thrpt}Mbps, Network: {net_thrpt}Mbps")
    return [net_score_value, io_score_value, len(tQueue)] #score_value


def normal_transfer(params):
    global network_throughput_logs, io_throughput_logs, file_processed, file_count
    if len(params) != 2:
        params = [2,2]

    logger.info("Normal Transfer -- Probing Parameters [Network, I/O]: {0}".format(params))

    for i in range(len(transfer_process_status)):
        transfer_process_status[i] = 1 if (i < params[0] and file_processed.value<file_count) else 0

    for i in range(len(io_process_status)):
        io_process_status[i] = 1 if (i < params[1] and file_copied.value<file_count) else 0

    logger.info(f"Transfer Process Status 649: {transfer_process_status}")
    logger.info(f"File Processed: {file_processed.value}, File Count: {file_count}")  

    while file_processed.value < file_count:
        time.sleep(0.1)

    logger.info("Exiting Write 652")


def run_optimizer(probing_func):
    global file_processed, file_count, network_throughput_logs, io_throughput_logs, tmpfs_dir
    params = [2,2]

    if configurations["mp_opt"]:
        if configurations["method"].lower() == "ppo":
            logger.info("Running PPO Optimization .... ")
            optimizer = PPOOptimizer()
            file_processed.value = file_count
            return
            # params = optimizer.ppo_probing(params)
        elif configurations["method"].lower() == "cg":
            logger.info("Running Conjugate Optimization .... ")
            params = cg_opt(configurations["mp_opt"], probing_func)

        elif configurations["method"].lower() == "mgd":
            logger.info("Running Multivariate Gradient Optimization .... ")
            params = gradient_multivariate(io_cc, net_cc, probing_func, logger)
        else:
            logger.info("Running Bayesian Optimization .... ")
            params = base_optimizer(configurations, probing_func, logger)

    else:
        cc_limit = net_cc if probing_func is network_probing else io_cc
        if configurations["method"].lower() == "hill_climb":
            logger.info("Running Hill Climb Optimization .... ")
            params = hill_climb(cc_limit, probing_func, logger)

        elif configurations["method"].lower() == "gradient":
            logger.info("Running Gradient Optimization .... ")
            params = gradient_opt_fast(cc_limit, probing_func, logger)

        elif configurations["method"].lower() == "cg":
            logger.info("Running Conjugate Optimization .... ")
            params = cg_opt(configurations["mp_opt"], probing_func)

        elif configurations["method"].lower() == "probe":
            logger.info("Running a fixed configurations Probing .... ")
            params = [configurations["fixed_probing"]["thread"], configurations["fixed_probing"]["thread"]]

        else:
            logger.info("Running Bayesian Optimization .... ")
            params = base_optimizer(configurations, probing_func, logger)


    if file_processed.value < file_count:
        normal_transfer(params)


def report_network_throughput(start_time):
    global network_throughput_logs, file_count, file_processed
    previous_total = 0
    previous_time = 0
    previous_transfer_file_offsets = None

    while file_processed.value < file_count:
        t1 = time.time()
        time_since_begining = np.round(t1-start_time, 1)

        if time_since_begining >= 0.1:
            total_bytes = np.sum(transfer_file_offsets)
            thrpt = np.round((total_bytes*8)/(time_since_begining*1000*1000), 2)

            curr_total = total_bytes - previous_total
            curr_time_sec = np.round(time_since_begining - previous_time, 3)
            curr_thrpt = np.round((curr_total*8)/(curr_time_sec*1000*1000), 2)
            network_throughput_logs.append(curr_thrpt)
            logger.info(f"Network Throughput @{time_since_begining}s, Current: {curr_thrpt}Mbps, Average: {thrpt}Mbps")
            previous_time, previous_total = time_since_begining, total_bytes
            t2 = time.time()
            fname = 'timed_log_network_ppo_' + configurations['model_version'] +'.csv'
            with open(fname, 'a') as f:
                f.write(f"{t2}, {time_since_begining}, {curr_thrpt}, {sum(transfer_process_status)}\n")
            time.sleep(max(0, 1 - (t2-t1)))


def report_io_throughput(start_time):
    global io_throughput_logs
    previous_total = 0
    previous_time = 0

    while file_processed.value < file_count:
        t1 = time.time()
        time_since_begining = np.round(t1-start_time, 1)

        if time_since_begining >= 0.1:
            total_bytes = np.sum(io_file_offsets)
            thrpt = np.round((total_bytes*8)/(time_since_begining*1000*1000), 2)
            curr_total = total_bytes - previous_total
            curr_time_sec = np.round(time_since_begining - previous_time, 3)
            curr_thrpt = np.round((curr_total*8)/(curr_time_sec*1000*1000), 2)
            previous_time, previous_total = time_since_begining, total_bytes
            io_throughput_logs.append(curr_thrpt)
            logger.info(f"I/O Throughput @{time_since_begining}s, Current: {curr_thrpt}Mbps, Average: {thrpt}Mbps")

            t2 = time.time()
            fname = 'timed_log_read_ppo_' + configurations['model_version'] +'.csv'
            with open(fname, 'a') as f:
                f.write(f"{t2}, {time_since_begining}, {curr_thrpt}, {sum(io_process_status)}\n")
            time.sleep(max(0, 1 - (t2-t1)))

from sender_helper import start_log_listener

def graceful_exit(signum=None, frame=None):
    logger.debug((signum, frame))
    try:
        rQueue.clear()
        tQueue.clear()
    except Exception as e:
        logger.debug(e)

    # fetch_logs_via_socket()
    logger.info("Exiting Sender....")
    host, port = configurations["sender"]["host"], configurations["sender"]["port"]
    start_log_listener(configurations, host=host, port=int(port))
    shutil.rmtree(tmpfs_dir, ignore_errors=True)
    exit(1)

def debug_concurrency():
    import threading
    logger.info("=== Threads ===")
    for t in threading.enumerate():
        logger.info(f" - {t.name} (alive={t.is_alive()}, daemon={t.daemon})")

    logger.info("=== Processes ===")
    for c in mp.active_children():
        logger.info(f" - {c.name} PID={c.pid} (alive={c.is_alive()})")


if __name__ == '__main__':
    signal.signal(signal.SIGINT, graceful_exit)
    signal.signal(signal.SIGTERM, graceful_exit)

    if os.path.exists('timed_log_network_ppo_' + configurations['model_version'] +'.csv'):
        os.remove('timed_log_network_ppo_' + configurations['model_version'] +'.csv')
    if os.path.exists('timed_log_read_ppo_' + configurations['model_version'] +'.csv'):
        os.remove('timed_log_read_ppo_' + configurations['model_version'] +'.csv')
    if os.path.exists('shared_memory_log_sender_ppo_' + configurations['model_version'] +'.csv'):
        os.remove('shared_memory_log_sender_ppo_' + configurations['model_version'] +'.csv')
    if not os.path.exists('logs/'):
        os.makedirs('logs/')

    net_cc = configurations["max_cc"]["network"]
    net_cc = net_cc if net_cc>0 else mp.cpu_count()


    io_cc = configurations["max_cc"]["io"]
    io_cc = io_cc if io_cc>0 else mp.cpu_count()
    configurations["thread_limit"] = max(net_cc, io_cc)

    log_FORMAT = '%(created)f -- %(levelname)s: %(message)s'
    log_file = f'logs/sender.{datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")}.log'

    if configurations["loglevel"] == "debug":
        logger.basicConfig(
            format=log_FORMAT,
            datefmt='%m/%d/%Y %I:%M:%S %p',
            level=logger.DEBUG,
            # filename=log_file,
            # filemode="w"
            handlers=[
                logger.FileHandler(log_file),
                logger.StreamHandler()
            ]
        )

        mp.log_to_stderr(logger.DEBUG)
    else:
        logger.basicConfig(
            format=log_FORMAT,
            datefmt='%m/%d/%Y %I:%M:%S %p',
            level=logger.INFO,
            # filename=log_file,
            # filemode="w"
            handlers=[
                logger.FileHandler(log_file),
                logger.StreamHandler()
            ]
        )

        # mp.log_to_stderr(logger.INFO)

    network_limit = -1
    if "network_limit" in configurations and configurations["network_limit"] is not None:
        network_limit = configurations["network_limit"]

    io_limit = -1
    if "io_limit" in configurations and configurations["io_limit"] is not None:
        io_limit = int(configurations["io_limit"])

    file_transfer = True

    manager = mp.Manager()
    root_dir = configurations["data_dir"]
    tmpfs_dir = f"/dev/shm/data{os.getpid()}/"
    probing_time = configurations["probing_sec"]
    file_names = [f"{fname}-{i}" for fname in os.listdir(root_dir) for i in range(1, configurations["multiplier"] + 1)]
    file_sizes = [os.path.getsize(os.path.join(root_dir, fname.split('-')[0])) for fname in file_names]
    file_count = len(file_names)
    network_throughput_logs = manager.list()
    io_throughput_logs = manager.list()
    exit_signal = 10 ** 10
    chunk_size = 1 * 1024 * 1024
    transfer_process_status = mp.Array("i", [0 for i in range(net_cc)])
    io_process_status = mp.Array("i", [0 for i in range(io_cc)])
    transfer_file_offsets = mp.Array("d", [0 for i in range(file_count)])
    io_file_offsets = mp.Array("d", [0 for i in range(file_count)])
    
    from multiprocessing import Value

    file_processed = Value('i', 0)
    file_copied = Value('i', 0)
    logger.info(f"File Count: {file_count} in {root_dir}")
    # io_weight, net_weight = 1, 1

    HOST, PORT = configurations["receiver"]["host"], configurations["receiver"]["port"]
    RCVR_ADDR = str(HOST) + ":" + str(PORT)

    send_file_count(HOST, PORT, file_count)

    try:
        os.mkdir(tmpfs_dir)
    except Exception as e:
        logger.debug(str(e))
        exit(1)

    _, free = available_space(tmpfs_dir)
    memory_limit = min(configurations["memory_use"]["maximum"], free/2)

    rQueue = manager.dict()
    tQueue = manager.dict()

    for i in range(file_count):
        rQueue[i] = 0

    copy_workers = [mp.Process(target=copy_file, args=(i,)) for i in range(io_cc)]
    for p in copy_workers:
        p.daemon = True
        p.start()

    transfer_workers = [mp.Process(target=transfer_file, args=(i,)) for i in range(net_cc)]
    for p in transfer_workers:
        p.daemon = True
        p.start()

    start = time.time()
    network_report_thread = Thread(target=report_network_throughput, args=(start,))
    network_report_thread.start()

    io_report_thread = Thread(target=report_io_throughput, args=(start,))
    io_report_thread.start()

    if configurations["mp_opt"]:
        optimizer_thread = Thread(target=run_optimizer, args=(multi_params_probing,))
        optimizer_thread.start()

    else:
        io_optimizer_thread = Thread(target=run_optimizer, args=(io_probing,))
        io_optimizer_thread.start()

        network_optimizer_thread = Thread(target=run_optimizer, args=(network_probing,))
        network_optimizer_thread.start()

    while file_processed.value < file_count:
        time.sleep(1)

    end = time.time()
    time_since_begining = np.round(end-start, 3)
    total = np.round(np.sum(file_sizes) / (1024*1024*1024), 3)
    thrpt = np.round((total*8*1024)/time_since_begining,2)
    logger.info("Total: {0} GB, Time: {1} sec, Throughput: {2} Mbps".format(
        total, time_since_begining, thrpt))

    for p in copy_workers:
        if p.is_alive():
            p.terminate()
            p.join(timeout=0.1)

    for p in transfer_workers:
        if p.is_alive():
            p.terminate()
            p.join(timeout=0.1)

    host, port = configurations["sender"]["host"], configurations["sender"]["port"]
    start_log_listener(configurations, host=host, port=int(port))
    
    logger.info(f'tmpfs_dir: {tmpfs_dir}')
    shutil.rmtree(tmpfs_dir, ignore_errors=True)
    debug_concurrency()
