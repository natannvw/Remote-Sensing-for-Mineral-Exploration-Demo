import time
import threading
import psutil
import os

class PerformanceMonitor:
    # perf_monitor = PerformanceMonitor()
    # perf_monitor.start()
    # my_function()
    # perf_monitor.stop()
    
    def __init__(self):
        self.memory_monitor = MemoryMonitorThread()
        self.start_time = None

    def start(self):
        self.memory_monitor.start_memory_monitoring()
        self.start_time = time.time()

    def stop(self):
        elapsed_time = time.time() - self.start_time
        print(f"Elapsed time: {elapsed_time} seconds")
        max_memory_usage = self.memory_monitor.stop_memory_monitoring()
        return elapsed_time, max_memory_usage
    
class MemoryMonitorThread(threading.Thread):
    def __init__(self, interval=0.1):
        super().__init__()
        self.interval = interval
        self.running = False
        self.result = 0

    def run(self):
        process = psutil.Process(os.getpid())
        max_memory_usage = 0

        while self.running:
            memory_usage = process.memory_info().rss / (1024 ** 2)  # Memory usage in MB
            max_memory_usage = max(max_memory_usage, memory_usage)
            time.sleep(self.interval)

        self.result = max_memory_usage

    def start_memory_monitoring(self):
        self.running = True
        self.start()

    def stop_memory_monitoring(self):
        self.running = False
        self.join()
        
        max_memory_usage = self.result
        print(f"Max memory usage: {max_memory_usage} MB")

        return self.result