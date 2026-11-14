"""Three-state Viterbi (home / away / dead) with per-frame transition costs, in C (solution/native/
viterbi3.c, compiled on first use with the system compiler), with a pure-numpy fallback.

decode(p3, blocks, cost_team, cost_in, cost_out): p3 (n, 3) probabilities; the costs are scalars or
(n,) arrays: cost_team for a home <-> away change, cost_in for alive -> dead, cost_out for dead ->
alive, each charged on the frame the new state starts. Frames outside the blocks are X (3).
With scalar costs and cost_in == cost_out it gives the same path as possession_ml.decode3.
"""

import ctypes
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "native", "viterbi3.c")
LIB = os.path.join(HERE, "native", "libviterbi3" + (".dylib" if os.uname().sysname == "Darwin" else ".so"))
_lib = None


def _load():
    global _lib
    if _lib is None:
        if not os.path.exists(LIB) or os.path.getmtime(LIB) < os.path.getmtime(SRC):
            subprocess.run(["cc", "-O2", "-shared", "-fPIC", "-o", LIB, SRC], check=True)
        lib = ctypes.CDLL(LIB)
        d = ctypes.POINTER(ctypes.c_double)
        lib.viterbi3.argtypes = [d, d, d, d, ctypes.c_long, ctypes.POINTER(ctypes.c_int8)]
        lib.viterbi3.restype = None
        _lib = lib
    return _lib


def _arr(c, n):
    a = np.broadcast_to(np.asarray(c, dtype=np.float64), (n,))
    return np.ascontiguousarray(a)


def _path(lp, ct, ci, co):
    n = len(lp)
    out = np.empty(n, dtype=np.int8)
    try:
        lib = _load()
        P = ctypes.POINTER(ctypes.c_double)
        lib.viterbi3(np.ascontiguousarray(lp, dtype=np.float64).ctypes.data_as(P), ct.ctypes.data_as(P),
                     ci.ctypes.data_as(P), co.ctypes.data_as(P), n, out.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)))
        return out
    except OSError:
        pass
    s = lp[0].copy()
    back = np.zeros((n, 3), dtype=np.int8)
    for t in range(1, n):
        T = np.array([[0.0, ct[t], ci[t]], [ct[t], 0.0, ci[t]], [co[t], co[t], 0.0]])
        cand = s[:, None] - T
        back[t] = np.argmax(cand, axis=0)
        s = cand[back[t], np.arange(3)] + lp[t]
    k = int(np.argmax(s))
    for t in range(n - 1, -1, -1):
        out[t] = k
        k = back[t, k]
    return out


def decode(p3, blocks, cost_team=16.0, cost_in=32.0, cost_out=None, emit=None):
    """H/A/D codes (X outside the blocks). emit: optional (n, 3) log-emissions instead of log p3."""
    n = len(p3)
    cost_out = cost_in if cost_out is None else cost_out
    lp = np.log(np.clip(p3, 1e-4, 1.0)) if emit is None else emit
    ct, ci, co = _arr(cost_team, n), _arr(cost_in, n), _arr(cost_out, n)
    out = np.full(n, 3, dtype=np.int8)
    for _, a, z in blocks:
        sl = slice(a, z + 1)
        out[sl] = _path(lp[sl], ct[sl], ci[sl], co[sl])
    return out
