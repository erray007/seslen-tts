# -*- coding: utf-8 -*-
"""
TAŞIYICI — kısa cümlede (model.KISA_SOZCUK) taşıyıcı seçenekleri: süre ve özdenetim tanıyıcısının duyduğu.

Kısa cümleye (1-2 sözcük) model sonda uydurma hece ekliyor ("Evet." → "evetnı"); model.TASIYICI cümleyi uzatıp son
harfte keser. Bedeli: çözücü taşıyıcıyı da üretir, ekran okuyucu ifadeleri ("Geri düğmesi.") 2-3 kat yavaşlar.

    python araclar/hiz/tasiyici.py [tekrar]       # gerçek okunuş (turkce-vurgu) ve denetim.onnx gerekir

Ölçü (seslen_tts.denetim): birebir = tanıyıcının duyduğu metinle aynı; kuyruk = sonda fazladan harf duyuldu; ceza =
Ozdenetim.dinle cezası (0 kusursuz). Süre: konus() ortalaması (ms). Eski deneme (taşıyıcıyı çözücüye vermeden
hizalamayı kırpmak, InstanceNorm yüzünden 6,9-8,3 dB bozuk) git geçmişinde.
"""
from __future__ import annotations

import sys
import time

import numpy as np

from ortak import M

from seslen_tts.denetim import Ozdenetim, Taniyici, normal   # noqa: E402

IFADELER = ["Evet.", "Tamam.", "Geri.", "Bitti.", "Katkı.", "Dikkat!", "Ayarlar.", "Kamera.", "Telefon.", "Mesajlar.",
            "Bağlandı.", "Kapalı.", "Açık.", "Kilitli.", "Ana ekran.", "Geri düğmesi.", "Ses düzeyi.", "Bildirimler.",
            "Gelen kutusu.", "Arama yap."]
SESLER = ["nehir", "ege", "ayse", "serdar"]
SECENEKLER = {                       # ad → taşıyıcı (None: taşıyıcısız, KISA_SOZCUK = 0)
    "tam (şimdiki)": M.TASIYICI,
    "yok": None,
    "hem de": " Hem de.",
    "hem bunu da": " Hem bunu da.",
    "hem bunu söyledim": " Hem bunu söyledim.",
    "tam, hızlı ×1/2": (M.TASIYICI, 0.5),
    "tam, hızlı ×1/3": (M.TASIYICI, 1 / 3),
    "tam, hızlı ×1/5": (M.TASIYICI, 0.2),
    "söyledim, hızlı ×1/3": (" Hem bunu söyledim.", 1 / 3),
}


def hizli_konus(sm: M.SesModeli, metin: str, stil: np.ndarray, kat: float | None) -> np.ndarray:
    """Taşıyıcının süresi kat ile çarpılır (son harf ve ardındaki noktalama olduğu gibi kalır)."""
    if not kat:
        return sm.konus(metin, stil)
    bitis = sm._kisa_bitis(metin)
    if not bitis:
        return sm.konus(metin, stil)
    n = sum(c in sm.semboller for c in sm.okunus(metin + M.TASIYICI))
    return sm.konus(metin, stil, uzat=[(bitis + 2, n + 1, kat)])


def dene(sm: M.SesModeli, oz: Ozdenetim, tasiyici, tekrar: int) -> tuple[float, float, float, float]:
    tasiyici, kat = tasiyici if isinstance(tasiyici, tuple) else (tasiyici, None)
    kisa, tas = M.KISA_SOZCUK, M.TASIYICI
    M.KISA_SOZCUK, M.TASIYICI = (0, tas) if tasiyici is None else (kisa, tasiyici)
    from turkce_vurgu import normallestir
    birebir = kuyruk = 0
    cezalar, sureler = [], []
    try:
        for ses in SESLER:
            stil = sm.stil(ses)
            for m in IFADELER:
                ref = normal(normallestir(m))
                for _ in range(tekrar):
                    t0 = time.perf_counter()
                    w = hizli_konus(sm, m, stil, kat)
                    sureler.append(time.perf_counter() - t0)
                    ceza, _, duyulan = oz.dinle(w, M.ORAN, ref)
                    d, r = duyulan.strip(), ref.strip()
                    birebir += d == r
                    kuyruk += d.startswith(r) and len(d) > len(r)
                    cezalar.append(ceza)
    finally:
        M.KISA_SOZCUK, M.TASIYICI = kisa, tas
    n = len(cezalar)
    return birebir / n, kuyruk / n, float(np.mean(cezalar)), float(np.mean(sureler)) * 1000


if __name__ == "__main__":
    tekrar = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    sm = M.SesModeli(None, 4)
    oz = Ozdenetim(sm, Taniyici(sm.klasor, 4))
    print(f"{len(SESLER)} ses × {len(IFADELER)} ifade × {tekrar} tekrar")
    for ad, t in SECENEKLER.items():
        b, k, c, s = dene(sm, oz, t, tekrar)
        print(f"{ad:20} birebir %{b * 100:4.0f}  kuyruk %{k * 100:4.0f}  ceza {c:5.2f}  süre {s:4.0f} ms")
