"""Capture the golden INT8 / MoE Metal kernels for the on-device compile probe (story 5.1).

Run with the golden venv (Apple M5 for the reference outputs):
    ../mlx-lm-as-classifiers/.venv/bin/python scripts/capture_int8_kernels.py OUT.json

`mx.fast.metal_kernel` is patched before the golden modules build their kernels, so
the source and header strings are exactly the golden ones (no hand port). Each kernel
is then called once through its golden wrapper on a small problem; the call's launch
geometry, template, input bytes and the M5 output bytes are recorded. The iOS probe
(App/ClassifierProbe, `--kernels`) rebuilds each kernel with `MLXFast.metalKernel`,
replays the call, and compares its outputs with the M5 bytes.
"""

import base64
import json
import sys

import mlx.core as mx

_real = mx.fast.metal_kernel
captured = []


def _bytes(a):
    a = mx.contiguous(a)
    mx.eval(a)
    return base64.b64encode(bytes(memoryview(a.view(mx.uint8).reshape(-1)))).decode()


def _dt(d):
    return str(d).replace("mlx.core.", "")


def recording_kernel(**kw):
    real = _real(**kw)

    def call(inputs, output_shapes, output_dtypes, grid, threadgroup, template=None, **rest):
        outs = real(inputs=inputs, output_shapes=output_shapes, output_dtypes=output_dtypes,
                    grid=grid, threadgroup=threadgroup, template=template, **rest)
        mx.eval(outs)
        captured.append({
            "name": kw["name"],
            "input_names": list(kw["input_names"]),
            "output_names": list(kw["output_names"]),
            "header": kw.get("header", ""),
            "source": kw["source"],
            "ensure_row_contiguous": kw.get("ensure_row_contiguous", True),
            "grid": list(grid), "threadgroup": list(threadgroup),
            "template": [[k, _dt(v) if isinstance(v, mx.Dtype) else v] for k, v in (template or [])],
            "inputs": [{"shape": list(a.shape), "dtype": _dt(a.dtype), "b64": _bytes(a)} for a in inputs],
            "outputs": [{"shape": list(o.shape), "dtype": _dt(o.dtype), "b64": _bytes(o)} for o in outs],
        })
        return outs

    return call


mx.fast.metal_kernel = recording_kernel

from mlx_lm.models.switch_layers import QuantizedSwitchLinear  # noqa: E402

from mlx_classifiers import moe  # noqa: E402
from mlx_classifiers.int8 import experts as ex  # noqa: E402
from mlx_classifiers.int8 import wiring  # noqa: E402

key = mx.random.key
# wiring.py: W8A8 GEMM (MPP matmul2d), row quantiser, int8 GEMV
M, N, K = 64, 128, 256
A = mx.random.randint(-127, 128, (M, K), key=key(1)).astype(mx.int8)
B = mx.random.randint(-127, 128, (N, K), key=key(2)).astype(mx.int8)
wiring._run_int8_linear(A, B, mx.random.uniform(shape=(M,), key=key(3)) * 0.01,
                        mx.random.uniform(shape=(N,), key=key(4)) * 0.01, M, N, K)
wiring._rowquant_int8((mx.random.normal((4, K), key=key(5)) * 3).astype(mx.float16))
wiring._gemv_int8(mx.random.normal((2, 512), key=key(6)).astype(mx.float16),
                  mx.random.randint(-127, 128, (16, 512), key=key(7)).astype(mx.int8),
                  mx.random.uniform(shape=(16,), key=key(8)) * 0.01, 16, 512)

# experts.py: W4A8 row quantiser + gather GEMM (MPP matmul2d, int4b), as test_int8_experts
experts, n, k, rows = 4, 64, 128, 16
w = mx.random.normal((experts, n, k), key=key(rows)) * 0.05 + 0.01
wq, s, b = mx.quantize(w.astype(mx.bfloat16), group_size=64, bits=4)
lin = QuantizedSwitchLinear(k, n, experts, bias=False)
lin.weight, lin.scales, lin.biases = wq, s, b
il = ex.Int8SwitchLinear(lin)
ids = mx.sort(mx.random.randint(0, experts - 1, (rows,), key=key(2))).astype(mx.int32)
q, xs, rs = ex.quant_rows(mx.random.normal((rows, k), key=key(1)).astype(mx.bfloat16))


def pad(a):
    return mx.concatenate([a, mx.zeros((ex.SM, *a.shape[1:]), a.dtype)])


ex.gather_gemm(pad(q), pad(xs), pad(rs), il, ex.build_tiles(ids, rows, experts))

# moe.py: segmented gather qmm (inlined MLX NAX headers), as moe._self_check
if moe.available():
    E, n2, k2, r2 = 8, 128, 256, 8 * 6
    w2 = (mx.random.normal((E, n2, k2), key=key(3)) * 0.05).astype(mx.bfloat16)
    wq2, s2, b2 = mx.quantize(w2, group_size=64, bits=4)
    x2 = mx.random.normal((r2, k2), key=key(4)).astype(mx.bfloat16)
    ids2 = mx.sort(mx.random.randint(0, E, (r2,), key=key(5))).astype(mx.uint32)
    moe.segmented_gather_qmm(x2, wq2, s2, b2, moe.expert_offsets(ids2, E))
else:
    print("moe.available() is False here; segmented kernel not captured", file=sys.stderr)

json.dump({"format": "mlx-classifiers-kernel-probe/1", "mlx": mx.__version__,
           "device": mx.device_info().get("architecture", ""), "kernels": captured},
          open(sys.argv[1], "w"))
print(f"captured {len(captured)} kernels: {[c['name'] for c in captured]}")
