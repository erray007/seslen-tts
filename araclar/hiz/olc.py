# -*- coding: utf-8 -*-
"""
HIZ — kısa / orta / uzun cümlede metin tarafı ve konus() süresi, RTF.

    python araclar/hiz/olc.py              # iş parçacığı: model varsayılanı (en çok 6)
    python araclar/hiz/olc.py 1 2 4        # her iş parçacığı sayısı için ayrı

06.11.2026, 4 çekirdek Xeon 2,3 GHz (varsayılan): kısa 1,4 sn ses 0,32 sn (RTF 0,23), orta 3,8 sn 0,78 sn (0,20),
uzun 6,3 sn 1,16 sn (0,18). Zamanın ~%90'ı çözücüde (ses.onnx).
"""
from __future__ import annotations

import statistics as st
import sys
import time

from ortak import kaba, ses_modeli

CUMLELER = {
    "kisa": "Ayarlar düğmesi.",
    "orta": "Yarın sabah saat dokuzda toplantımız var, dosyaları hazırladım.",
    "uzun": "Bir varmış bir yokmuş, ormanın derinliklerinde meraklı ve çok çalışkan bir sincap yaşarmış, her sabah "
            "erkenden kalkıp fındık toplarmış.",
}


def olc(is_parcacigi: int, n: int = 5) -> None:
    t0 = time.perf_counter()
    sm = ses_modeli(is_parcacigi)
    print(f"açılış {time.perf_counter() - t0:.2f} sn, iş parçacığı {is_parcacigi or 'varsayılan'}")
    stil = sm.stil("nehir")
    for ad, c in CUMLELER.items():
        ok = kaba(c)
        sm.konus(c, stil, okunus=ok)                        # ısınma
        tm, tk, sure = [], [], 0.0
        for _ in range(n):
            t0 = time.perf_counter()
            sm._metin_tarafi(ok, stil.reshape(1, -1))
            t1 = time.perf_counter()
            w = sm.konus(c, stil, okunus=ok)
            tm.append(t1 - t0)
            tk.append(time.perf_counter() - t1)
            sure = len(w) / 24000
        k = st.median(tk)
        print(f"  {ad:5} ses {sure:5.2f} sn  metin {st.median(tm) * 1000:5.0f} ms  konus {k * 1000:6.0f} ms  "
              f"RTF {k / sure:.3f}")


if __name__ == "__main__":
    for th in [int(a) for a in sys.argv[1:]] or [0]:
        olc(th)
