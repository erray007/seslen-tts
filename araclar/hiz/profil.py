# -*- coding: utf-8 -*-
"""
PROFİL — ses.onnx'te zamanın işlem türlerine dağılımı (ORT profilleyicisi, uzun cümle, 3 çalıştırma).

    python araclar/hiz/profil.py [ses.onnx yolu]

06.11.2026 (4 iş parçacığı): Conv %57, Sin %19 (Snake etkinleştirmesi; ORT'de Sin tek iş parçacığında, bkz.
mikro.py), InstanceNormalization %6, ConvTranspose %5, Mul %5, Add %4.
"""
from __future__ import annotations

import collections
import json
import os
import sys

from ortak import cozucu_girdisi, model_klasoru, oturum, ses_modeli
from olc import CUMLELER

if __name__ == "__main__":
    yol = sys.argv[1] if len(sys.argv) > 1 else model_klasoru() / "ses.onnx"
    g = cozucu_girdisi(ses_modeli(), CUMLELER["uzun"])
    s = oturum(yol, enable_profiling=True, profile_file_prefix="profil")
    for _ in range(3):
        s.run(None, g)
    dosya = s.end_profiling()
    sure, tane = collections.Counter(), collections.Counter()
    for e in json.load(open(dosya)):
        if e.get("cat") == "Node" and e["name"].endswith("_kernel_time"):
            sure[e["args"]["op_name"]] += e["dur"]
            tane[e["args"]["op_name"]] += 1
    os.remove(dosya)
    top = sum(sure.values())
    for ad, v in sure.most_common(12):
        print(f"{ad:24} %{v / top * 100:5.1f}  {v / 3 / 1000:7.1f} ms/çalıştırma  ({tane[ad] // 3} düğüm)")
