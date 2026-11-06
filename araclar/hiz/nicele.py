# -*- coding: utf-8 -*-
"""
NİCELE — çözücünün seçilen Conv'larını int8'e indirir (QDQ, kanal başına ağırlık, MinMax kalibrasyonu
cikti/cal_*.npz). Girdi cikti/sabit_on.onnx (yama.py sabit + ORT ön işleme; yoksa kurulur).

    python araclar/hiz/nicele.py resblocks.0,resblocks.1,resblocks.2,noise_res cikti/q_guvenli.onnx

06.11.2026, 16 sınama girdisi (kıyas özgün ↔ özgün 0,52 dB; özgün 10,1 sn):
  ilk aşama (resblocks 0-2) + noise_res   0,55 dB   7,9 sn  ×1,28   ← güvenli
  + resblocks.3                           0,63 dB   7,3 sn  ×1,38
  + resblocks.4 ya da .5                  0,81-0,85 dB
  bütün resblock + noise_res              1,12 dB   6,2 sn  ×1,63
  bütün Conv (tek cümle kalibrasyon)      6,52 dB   bozuk
Son aşama (resblocks 3-5, dalgaya en yakın) nicelemeye hassas. Elenenler: dinamik nicelemede Conv (ConvInteger)
7 kat yavaş; yalnız Gemm/MatMul hız getirmiyor, 1,18 dB. Percentile kalibrasyonu 15 GB'ı aşıyor.
"""
from __future__ import annotations

import sys

import numpy as np
import onnx
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                      quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

from ortak import CIKTI
from yama import yamala


class _Okur(CalibrationDataReader):
    def __init__(self):
        self.v = iter([dict(np.load(f)) for f in sorted(CIKTI.glob("cal_*.npz"))])

    def get_next(self):
        return next(self.v, None)


def nicele(desenler: list[str], cikis: str) -> int:
    on = CIKTI / "sabit_on.onnx"
    if not on.exists():
        if not (CIKTI / "sabit.onnx").exists():
            yamala(["sabit"])
        quant_pre_process(str(CIKTI / "sabit.onnx"), str(on), skip_symbolic_shape=True)
    m = onnx.load(str(on), load_external_data=False)
    secilen = [n.name for n in m.graph.node if n.op_type == "Conv" and any(d in n.name for d in desenler)]
    quantize_static(str(on), cikis, _Okur(), quant_format=QuantFormat.QDQ, nodes_to_quantize=secilen,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True,
                    calibrate_method=CalibrationMethod.MinMax)
    return len(secilen)


if __name__ == "__main__":
    print(sys.argv[2], nicele(sys.argv[1].split(","), sys.argv[2]), "Conv")
