import os
import sys
import runpy

if __name__ == "__main__":
    src_bench = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "benchmark_all.py")
    runpy.run_path(src_bench, run_name="__main__")
