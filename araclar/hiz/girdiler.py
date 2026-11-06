# -*- coding: utf-8 -*-
"""
GİRDİLER — çözücünün kalibrasyon (cikti/cal_*.npz, 16) ve sınama (cikti/test_*.npz, 16) girdileri: farklı ses,
ton ve cümle (ekran okuyucu ifadeleri, bildirim, anlatı).

    python araclar/hiz/girdiler.py
"""
from __future__ import annotations

import numpy as np

from ortak import CIKTI, cozucu_girdisi, ses_modeli

KALIBRASYON = ["Merhaba, nasılsın bugün?", "Bildirim: üç yeni mesajınız var.", "Pil seviyesi yüzde on beş, şarj edin.",
               "Ayarlar menüsü açıldı, aşağı kaydırın.", "Hava bugün güneşli ve sıcak olacak.",
               "Bu düğmeye iki kez dokunun.", "Ekran kilitlendi.",
               "Kitabın üçüncü bölümünde kahraman yola çıkıyor ve dağları aşıyor."]
SINAMA = ["Yarın sabah saat dokuzda toplantımız var, dosyaları hazırladım.", "Wi-Fi bağlantısı kesildi, yeniden deneniyor.",
          "Bir varmış bir yokmuş, ormanda meraklı bir sincap yaşarmış.", "Gelen arama: annem."]
TONLAR = ["notr", "neseli", "uzgun", "fisilti", "heyecan", "merak", "uyari", "kizgin"]

if __name__ == "__main__":
    sm = ses_modeli()
    sesler = list(sm.sesler)
    for i, c in enumerate(KALIBRASYON * 2):
        np.savez(CIKTI / f"cal_{i:02}.npz", **cozucu_girdisi(sm, c, sesler[i % len(sesler)], TONLAR[i % len(TONLAR)]))
    for i, c in enumerate(SINAMA):
        for j, ses in enumerate(sesler[::4]):
            np.savez(CIKTI / f"test_{i}_{j}.npz", **cozucu_girdisi(sm, c, ses, "notr" if j % 2 == 0 else "neseli"))
    print(f"{len(sesler)} ses; kalibrasyon {len(KALIBRASYON) * 2}, sınama {len(SINAMA) * 4} girdi → {CIKTI}")
