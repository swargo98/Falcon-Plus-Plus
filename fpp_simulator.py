from config_sender import configurations
from queue import PriorityQueue
import numpy as np
import random

class SimulatorState:
    def __init__(self, network_throughput=0, network_thread=0) -> None:
        # WILL TRY WITH only the current thread counts and throughputs, and then with window of history and then with gradient
        self.network_throughput = network_throughput
        self.network_thread = network_thread

    def copy(self):
        # Return a new SimulatorState instance with the same attribute values
        return SimulatorState(
            network_throughput=self.network_throughput,
            network_thread=self.network_thread
        )

    def to_array(self):
        # Convert the state to a NumPy array
        return np.array([
            self.network_throughput,
            self.network_thread
        ], dtype=np.float32)

class NetworkSystemSimulator:
    def __init__(self, network_thread = 1, network_throughput_per_thread = 2, network_bandwidth = 6, network_background_traffic = 0, track_states = False):
        self.network_throughput_per_thread = network_throughput_per_thread
        self.network_bandwidth = network_bandwidth
        self.network_background_traffic = network_background_traffic
        self.network_thread = network_thread
        self.track_states = track_states
        self.K = configurations['K']

    def network_thread_task(self, time, remaining_bw_per_thread):
        throughput_increase = 0
        # print(f"Network Thread start: Network Throughput: {throughput_increase}, Sender Buffer: {self.sender_buffer_in_use}, Receiver Buffer: {self.receiver_buffer_in_use}")

        random_float = random.uniform(0, 1)
        file_size = configurations['max_file_chunk_in_MB']
        if random_float < 0.9:
            file_size = configurations['max_file_chunk_in_MB'] / random.uniform(2,4)
        else:
            file_size = configurations['max_file_chunk_in_MB'] / random.uniform(4,8)

        network_throughput_temp = min(file_size, remaining_bw_per_thread)
        throughput_increase = min(network_throughput_temp, self.network_bandwidth-self.network_throughput)
        self.network_throughput += throughput_increase
        # print(f"Remaining BW per thread before allocation: {remaining_bw_per_thread}, Throughput Increase: {throughput_increase}, Network Throughput: {self.network_throughput}, Time: {time}")
        remaining_bw_per_thread -= throughput_increase
        # print(f"Remaining BW per thread after allocation: {remaining_bw_per_thread}")

        time_taken = throughput_increase / self.network_throughput_per_thread
        next_time = time + time_taken + 0.01
        if next_time < 1:
            self.thread_queue.put((next_time, remaining_bw_per_thread))
        # print(f"Network Thread end: Network Throughput: {throughput_increase}, Sender Buffer: {self.sender_buffer_in_use}, Receiver Buffer: {self.receiver_buffer_in_use}")
        # if throughput_increase > 0 and self.track_states:
        #     with open('thread_level_states.csv', 'a') as f:
        #         f.write(f"Network, {throughput_increase}, {self.sender_buffer_in_use}, {self.receiver_buffer_in_use}\n")
        return next_time

    def get_utility_value(self, threads):
        network_thread = threads[0]
        self.network_thread = network_thread

        self.thread_queue = PriorityQueue() # Key: time, Value: thread_type
        self.network_throughput = 0

        
        for i in range(network_thread):
            self.thread_queue.put((0, self.network_throughput_per_thread))

        network_thread_finish_time = 0

        while not self.thread_queue.empty():
            time, remaining_bw_per_thread = self.thread_queue.get()
            network_thread_finish_time = self.network_thread_task(time, remaining_bw_per_thread)

        self.network_throughput = self.network_throughput / network_thread_finish_time

        utility = (self.network_throughput/self.K ** network_thread)

        # print(f"Read thread: {read_thread}, Network thread: {network_thread}, Write thread: {write_thread}, Utility: {utility}")

        if self.track_states:
            with open('threads_throughputs'+ configurations['model_version'] +'.csv', 'a') as f:
                f.write(f"{network_thread}, {self.network_throughput}\n")

        final_state = SimulatorState(self.network_throughput, network_thread)

        return utility, final_state

#write a sample test case for the above class
if __name__ == "__main__":
    simulator = NetworkSystemSimulator(network_thread=2, network_throughput_per_thread=2000, network_bandwidth=6000, track_states=True)
    utility, final_state = simulator.get_utility_value([2])
    print(f"Utility: {utility}")
    print(f"Final State: {final_state.to_array()}")