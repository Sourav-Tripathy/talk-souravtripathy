
import os
import time
import torch

class MemoryTracker:
    def __init__(self, label: str, func_name: str):
        self.label = label
        self.func_name = func_name
        self.start_time = None
        self.ram_before = None
        self.vram_before = None

    def get_rss_mb(self) -> float:
        """Return current process RSS in MB (reads /proc/self/status)."""
        try:
            with open("/proc/self/status", "r") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return float(line.split()[1]) / 1024.0
        except Exception:
            pass
        return 0.0

    def print_memory_stats(self, checkpoint: str):
        print("\n" + "=" * 60)
        print(f"[{self.func_name}] -> {self.label} ({checkpoint})")
        print("-" * 60)
        print(f"[CPU] Current RSS Memory: {self.get_rss_mb():.2f} MB")

        if torch.cuda.is_available():
            gpu_alloc = torch.cuda.memory_allocated() / (1024 * 1024)
            gpu_res   = torch.cuda.memory_reserved()   / (1024 * 1024)
            print(f"[GPU] Memory Allocated:   {gpu_alloc:.2f} MB")
            print(f"[GPU] Memory Reserved:    {gpu_res:.2f} MB")
            try:
                free_mem, total_mem = torch.cuda.mem_get_info()
                print(f"[GPU] Free Memory:        {free_mem / (1024*1024):.2f} MB")
                print(f"[GPU] Total VRAM:         {total_mem / (1024*1024):.2f} MB")
            except Exception:
                pass
        else:
            print("[GPU] CUDA not available — running in CPU mode")
        print("=" * 60 + "\n")

    def __enter__(self):
        self.start_time = time.perf_counter()
        self.ram_before = self.get_rss_mb()
        self.vram_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        self.print_memory_stats("START")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (time.perf_counter() - self.start_time) * 1000
        ram_after = self.get_rss_mb()
        vram_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        status = "SUCCESS" if exc_type is None else "CRASH"
        self.print_memory_stats(f"END - {status}")
        
        ram_delta = ram_after - self.ram_before
        vram_delta = (vram_after - self.vram_before) / (1024 * 1024)
        print(f"[{self.func_name}] Duration: {elapsed:.2f} ms | RAM Delta: {ram_delta:.2f} MB | VRAM Delta: {vram_delta:.2f} MB")
