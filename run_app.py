import os
import sys
import runpy

if __name__ == "__main__":
    src_run = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "run_app.py")
    runpy.run_path(src_run, run_name="__main__")
