"""Builds (once) and loads mps_shared.mm, which gives MPS tensors whose bytes the CPU can also read and write.
Compiled with clang++ (Xcode or its Command Line Tools) into gguf/build/, named after the torch and Python
versions, so upgrading either rebuilds it."""
import ctypes
import importlib.util
import os
import subprocess
import sys
import sysconfig

import torch

NAME = "mps_shared"
HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, NAME + ".mm")
_module = None

def build(path: str):
    from torch.utils import cpp_extension
    lib = cpp_extension.library_paths()[0]
    cmd = ["clang++", "-std=c++17", "-O2", "-fobjc-arc", "-x", "objective-c++", "-shared",
           "-undefined", "dynamic_lookup", *["-I" + p for p in cpp_extension.include_paths()],
           "-I" + sysconfig.get_paths()["include"], f"-DTORCH_EXTENSION_NAME={NAME}",
           "-DTORCH_API_INCLUDE_EXTENSION_H", "-L" + lib, "-lc10", "-ltorch", "-ltorch_cpu", "-ltorch_python",
           "-Wl,-rpath," + lib, "-framework", "Metal", "-framework", "Foundation", SOURCE, "-o", path]
    print("building the GPU shared-memory helper (once)...", flush=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError("clang++ not found; install Xcode's Command Line Tools (`xcode-select --install`), "
                           "or rerun `main.py pack --no-gpu` to run experts on the CPU") from None
    if result.returncode:
        errors = "\n".join(line for line in result.stderr.splitlines() if "error" in line)
        raise RuntimeError(f"building {SOURCE} failed:\n{errors or result.stderr[-2000:]}")

def load():
    global _module
    if _module is None:
        tag = f"torch{torch.__version__.replace('+', '_')}-{sys.implementation.cache_tag}"
        path = os.path.join(HERE, "build", f"{NAME}-{tag}{sysconfig.get_config_var('EXT_SUFFIX')}")
        if not os.path.exists(path) or os.path.getmtime(path) < os.path.getmtime(SOURCE):
            build(path)
        spec = importlib.util.spec_from_file_location(NAME, path)
        _module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_module)
    return _module

def shared_empty(nbytes: int):
    """(uint8 MPS tensor of nbytes, writable memoryview of the same bytes on the CPU). Keep the tensor alive as
    long as the memoryview is used: the memory is freed with it."""
    tensor, address = load().shared_empty(nbytes)
    return tensor, memoryview((ctypes.c_ubyte * nbytes).from_address(address)).cast("B")
