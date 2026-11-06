# -*- coding: utf-8 -*-
"""
TAŞIYICI — kısa cümlede (model.KISA_SOZCUK) taşıyıcının çözücüye verilmemesi: hizalama son harf + pay birimde
kesilirse süre ve tam taşıyıcılı okumaya mel farkı (6 ekran okuyucu ifadesi × 3 ses).

    python araclar/hiz/tasiyici.py      # önce: python araclar/hiz/yama.py sabit

06.11.2026: tam 645 ms; pay 20 → 276 ms ama mel farkı 6,9 dB (en kötü 10,6), pay 2 → 203 ms, 8,3 dB. ELENDİ:
çözücüdeki InstanceNormalization (65) istatistiği bütün dalga üstünden alıyor; sonu kesilince başı da değişiyor.
"""
from __future__ import annotations

import time

import numpy as np

from mel import fark
from ortak import CIKTI, M, kaba, oturum, ses_modeli

KISA = ["Ayarlar düğmesi.", "Evet.", "Geri.", "Ekran kilitlendi.", "Mesajlar.", "Tamam."]


def calis(sm, cozucu, metin: str, ses: str, pay: int | None):
    stil = sm.stil(ses).reshape(1, -1)
    nokta = {sm.semboller[x] for x in ".!"}
    kod_k = [sm.semboller[x] for x in kaba(metin) if x in sm.semboller]
    son = len(kod_k)
    while son > 1 and kod_k[son - 1] in nokta:
        son -= 1
    _, d, t_en, sure = sm._metin_tarafi(kaba(metin + M.TASIYICI), stil)
    sure = np.maximum(np.round(sure), 1).astype(np.int64)
    sure[0] = min(sure[0], M.BAS_DOLGU)
    kes = int(sure[:son + 1].sum()) + 1                       # son harften 1 birim sonra (konus gibi)
    hiza = M.hizala(sure)
    if pay is not None:
        hiza = hiza[:, :, :min(hiza.shape[2], kes + pay)]
    g = {"d": d, "t_en": t_en, "hiza": hiza, "s": stil[:, 128:], "ref": stil[:, :128],
         "perde": np.array([1.], np.float32), "tonlama": np.array([1.], np.float32),
         "egri": np.zeros((1, len(sure)), np.float32)}
    t0 = time.perf_counter()
    w = cozucu.run(None, g)[0].reshape(-1)
    return w[:kes * M.ORNEK_BIRIM], time.perf_counter() - t0


if __name__ == "__main__":
    sm, cozucu = ses_modeli(4), oturum(CIKTI / "sabit.onnx")
    for pay in [None, 20, 10, 5, 2]:
        f, t = [], []
        for c in KISA:
            for ses in ["nehir", "ayse", "serdar"]:
                tam, _ = calis(sm, cozucu, c, ses, None)
                w, s = calis(sm, cozucu, c, ses, pay)
                f.append(fark(tam, w))
                t.append(s)
        print(f"pay {str(pay):5} çözücü {np.mean(t) * 1000:5.0f} ms   mel farkı {np.mean(f):.2f} (en kötü {max(f):.2f})")
