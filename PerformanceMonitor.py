import os
import threading
import time

import psutil


class PerformanceMonitor:
    """
    A class that monitors the performance of a program by measuring elapsed time and memory usage.

    Usage:
    perf_monitor = PerformanceMonitor()
    perf_monitor.start()
    my_function()
    perf_monitor.stop()
    """

    def __init__(self):
        """
        Initializes a PerformanceMonitor object.
        """
        self.memory_monitor = MemoryMonitorThread()
        self.start_time = None

    def start(self):
        """
        Starts the performance monitoring.
        """
        self.memory_monitor.start_memory_monitoring()
        self.start_time = time.time()

    def stop(self):
        """
        Stops the performance monitoring and returns the elapsed time and maximum memory usage.

        Returns:
        - elapsed_time (float): Elapsed time in seconds.
        - max_memory_usage (float): Maximum memory usage in MB.
        """
        elapsed_time = time.time() - self.start_time
        print(f"Elapsed time: {elapsed_time} seconds")
        max_memory_usage = self.memory_monitor.stop_memory_monitoring()
        return elapsed_time, max_memory_usage


class MemoryMonitorThread(threading.Thread):
    """
    A thread that monitors the memory usage of a program.

    Attributes:
    - interval (float): The interval between memory usage checks in seconds.
    - running (bool): Indicates whether the memory monitoring is running.
    - result (float): The maximum memory usage in MB.
    """

    def __init__(self, interval=0.1):
        """
        Initializes a MemoryMonitorThread object.

        Args:
        - interval (float): The interval between memory usage checks in seconds.
        """
        super().__init__()
        self.interval = interval
        self.running = False
        self.result = 0

    def run(self):
        """
        Runs the memory monitoring.
        """
        process = psutil.Process(os.getpid())
        max_memory_usage = 0

        while self.running:
            memory_usage = process.memory_info().rss / (1024**2)  # Memory usage in MB
            max_memory_usage = max(max_memory_usage, memory_usage)
            time.sleep(self.interval)

        self.result = max_memory_usage

    def start_memory_monitoring(self):
        """
        Starts the memory monitoring.
        """
        self.running = True
        self.start()

    def stop_memory_monitoring(self):
        """
        Stops the memory monitoring and returns the maximum memory usage.

        Returns:
        - max_memory_usage (float): Maximum memory usage in MB.
        """
        self.running = False
        self.join()

        max_memory_usage = self.result
        print(f"Max memory usage: {max_memory_usage} MB")

        return self.result
