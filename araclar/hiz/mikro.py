# -*- coding: utf-8 -*-
"""
MİKRO — ORT'de Sin ile Mul'un iş parçacığına göre süresi (1×128×75000 float32).

    python araclar/hiz/mikro.py

06.11.2026: Sin 1 iş parçacığında 34 ms, 4'te 33 ms (paralel değil); Mul 15,5 → 5,8 ms. Snake (x + sin²(αx)/α)
çözücü süresinin %19'u. Sin yerine Mul/Add polinomu (yama.py poli) doğru (SNR 84 dB) ama %6 yavaş: 13 küçük işlem
tek Sin'den pahalı. Kalıcı çözüm kaynaştırılmış özel çekirdek (C++ custom op).
"""
from __future__ import annotations

import time

import numpy as np
import onnxruntime as ort
from onnx import TensorProto as T
from onnx import helper as h


def model(dugum: str) -> bytes:
    g = h.make_graph([h.make_node(dugum, ["x"] if dugum == "Sin" else ["x", "x"], ["y"])], dugum,
                     [h.make_tensor_value_info("x", T.FLOAT, None)], [h.make_tensor_value_info("y", T.FLOAT, None)])
    return h.make_model(g, opset_imports=[h.make_opsetid("", 17)], ir_version=8).SerializeToString()


if __name__ == "__main__":
    x = (np.random.randn(1, 128, 75000) * 3).astype(np.float32)
    for th in (1, 4):
        so = ort.SessionOptions()
        so.intra_op_num_threads = th
        for d in ("Sin", "Mul"):
            s = ort.InferenceSession(model(d), so, providers=["CPUExecutionProvider"])
            s.run(None, {"x": x})
            t0 = time.perf_counter()
            for _ in range(5):
                s.run(None, {"x": x})
            print(f"iş parçacığı {th}  {d}: {(time.perf_counter() - t0) / 5 * 1000:.1f} ms")
