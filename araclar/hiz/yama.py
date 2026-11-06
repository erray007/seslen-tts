# -*- coding: utf-8 -*-
"""
YAMA — ses.onnx'in karşılaştırma kopyaları (cikti/ altına).

    python araclar/hiz/yama.py sabit            # cikti/sabit.onnx: kaynak gürültüsü (RandomNormalLike) sıfır
    python araclar/hiz/yama.py sabit poli       # cikti/sabit_poli.onnx: + Snake'in Sin'i polinom

sabit: iki çözücünün aynı girdide dalgaları karşılaştırılabilsin diye (gürültü kaliteyi değiştirmiyor: mel farkı
özgün ↔ özgün ile aynı, 0,51 dB). nicele.py bunun üstünde çalışır.
poli: sin r ≈ r + c3 r³ + c5 r⁵ + c7 r⁷, r = x − π·round(x/π) (kare alındığı için işaret önemsiz). 06.11.2026: SNR
84 dB, ama %6 yavaş (bkz. mikro.py).
"""
from __future__ import annotations

import sys

import numpy as np
import onnx
from onnx import helper as h
from onnx import numpy_helper as nh

from ortak import CIKTI, model_klasoru

C3, C5, C7 = -0.16666654611, 0.0083321608736, -0.00019587841


def _sabit(ad: str, v: float):
    return h.make_node("Constant", [], [ad], value=nh.from_array(np.array(v, np.float32)))


def _poli(n, p: str) -> list:
    x, m = n.input[0], h.make_node
    return [_sabit(p + "ip", 1 / np.pi), _sabit(p + "pi", np.pi), _sabit(p + "c3", C3), _sabit(p + "c5", C5),
            _sabit(p + "c7", C7), _sabit(p + "1", 1.0),
            m("Mul", [x, p + "ip"], [p + "a"]), m("Round", [p + "a"], [p + "q"]), m("Mul", [p + "q", p + "pi"], [p + "qp"]),
            m("Sub", [x, p + "qp"], [p + "r"]), m("Mul", [p + "r", p + "r"], [p + "u"]),
            m("Mul", [p + "u", p + "c7"], [p + "t1"]), m("Add", [p + "t1", p + "c5"], [p + "t2"]),
            m("Mul", [p + "t2", p + "u"], [p + "t3"]), m("Add", [p + "t3", p + "c3"], [p + "t4"]),
            m("Mul", [p + "t4", p + "u"], [p + "t5"]), m("Add", [p + "t5", p + "1"], [p + "t6"]),
            m("Mul", [p + "t6", p + "r"], [n.output[0]])]


def yamala(kipler: list[str]) -> str:
    m = onnx.load(str(model_klasoru() / "ses.onnx"))
    yeni, n_poli = [], 0
    for n in m.graph.node:
        if "sabit" in kipler and n.op_type == "RandomNormalLike":
            yeni.append(h.make_node("Sub", [n.input[0], n.input[0]], [n.output[0]], name=n.name))
        elif "poli" in kipler and n.op_type == "Sin" and ("resblocks" in n.name or "noise_res" in n.name):
            n_poli += 1
            yeni += _poli(n, f"_p{n_poli}_")
        else:
            yeni.append(n)
    del m.graph.node[:]
    m.graph.node.extend(yeni)
    yol = CIKTI / ("_".join(kipler) + ".onnx")
    onnx.save(m, str(yol))
    return f"{yol} (polinom: {n_poli})"


if __name__ == "__main__":
    print(yamala(sys.argv[1:] or ["sabit"]))
