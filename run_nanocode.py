import os
import sys
import runpy
from pathlib import Path


NANOCODE_ROOT = Path(__file__).resolve().parent

# Make NanoCode's packages take priority over workspace modules.
if str(NANOCODE_ROOT) in sys.path:
    sys.path.remove(str(NANOCODE_ROOT))

sys.path.insert(0, str(NANOCODE_ROOT))

runpy.run_module("cli.app", run_name="__main__")