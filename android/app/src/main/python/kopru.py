# -*- coding: utf-8 -*-
"""
KÖPRÜ — seslen_tts'i Android'de çalıştırır. Bütün mantık (okunuş, hizalama, soru ezgisi, duraklamalar, kısa cümle
taşıyıcısı, hız dengesi) seslen_tts'in kendisi; yalnız ONNX oturumları Android'in ONNX Runtime'ına (OrtKopru.kt)
yönlenir: Python'un onnxruntime paketi Android'de yok.

Kotlin şunları çağırır (Motor.kt): hazirla(klasor, kopru, is_parcacigi), isit(ses, ton),
seslendir(metin, ses, ton, hiz, perde) → yineleyici (her adımda cümle, dalga, sessizlik, ton, süre, aşamalar).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from seslen_tts import model as M
from seslen_tts.seslen import Seslen

_TUR = {np.dtype(np.float32): 0, np.dtype(np.int64): 1}
_SES = None
_HIZ_DOSYASI: Path | None = None
_SURE: dict[str, float] = {}           # aşama süreleri (ms), her cümlede sıfırlanır
# HIZLI TAŞIYICI: kısa cümlede (model.KISA_SOZCUK) taşıyıcının süresi bu katla çarpılır (1 = olduğu gibi). Taşıyıcı
# metinde kalır, çözücü daha az üretir. 06.11.2026 (araclar/hiz/tasiyici.py, 4 ses × 20 ifade): ×1/2 birebir %86 /
# 475 ms, olduğu gibi %89 / 681 ms (tek tekrar).
TASIYICI_KAT = 1.0


class _Oturum:
    """onnxruntime.InferenceSession yerine: run(None, girdiler) → [çıkışlar] (numpy)."""

    def __init__(self, kopru, ad: str):
        self.kopru, self.ad = kopru, ad

    def run(self, _cikislar, girdiler: dict):
        adlar, sekiller, turler, veriler = [], [], [], []
        for ad, d in girdiler.items():
            d = np.ascontiguousarray(d)
            adlar.append(ad)
            sekiller.append([int(x) for x in d.shape])
            turler.append(_TUR[d.dtype])
            veriler.append(d.tobytes())
        t0 = time.perf_counter()
        sonuc = self.kopru.calistir(self.ad, adlar, sekiller, turler, veriler)
        _SURE[self.ad] = _SURE.get(self.ad, 0.0) + (time.perf_counter() - t0) * 1000
        cikis = []
        for i in range(sonuc.sayi()):
            tur = np.float32 if sonuc.tur(i) == 0 else np.int64
            cikis.append(np.frombuffer(bytes(sonuc.veri(i)), tur).reshape([int(x) for x in sonuc.sekil(i)]))
        return cikis


class _AndroidModeli(M.SesModeli):
    def __init__(self, klasor: str, kopru, is_parcacigi: int):
        k = self.klasor = Path(klasor)
        self.is_parcacigi = is_parcacigi
        kopru.ac("metin", str(k / "metin.onnx"), is_parcacigi)
        kopru.ac("ses", str(k / "ses.onnx"), is_parcacigi)
        self.metin, self.ses = _Oturum(kopru, "metin"), _Oturum(kopru, "ses")
        v = np.load(k / "stiller.npz")
        self.stiller = {a: v[a].astype(np.float32) for a in v.files}
        self.sesler = json.loads((k / "sesler.json").read_text(encoding="utf-8"))
        self.semboller = json.loads((k / "semboller.json").read_text(encoding="utf-8"))
        self._virgul = {self.semboller[c] for c in "," if c in self.semboller}
        self._noktali = {self.semboller[c] for c in ";" if c in self.semboller}
        self._iki_nokta = {self.semboller[c] for c in ":" if c in self.semboller}

    def konus(self, metin, stil, *a, okunus=None, uzat=None, **kw):
        if TASIYICI_KAT != 1.0 and okunus is None and uzat is None:
            bitis = self._kisa_bitis(metin)
            if bitis:
                n = sum(c in self.semboller for c in self.okunus(metin + M.TASIYICI))
                uzat = [(bitis + 2, n + 1, TASIYICI_KAT)]
        return super().konus(metin, stil, *a, okunus=okunus, uzat=uzat, **kw)

    def okunus(self, metin: str) -> str:
        t0 = time.perf_counter()
        r = super().okunus(metin)
        _SURE["okunus"] = _SURE.get("okunus", 0.0) + (time.perf_counter() - t0) * 1000
        return r


def hazirla(klasor: str, kopru, is_parcacigi: int = 4) -> float:
    """Modeli açar, okunuş çeviricisini ısıtır; süre (sn). Sesin doğal hızları (Seslen.dogal_hiz) klasördeki
    dogal_hiz.json'dan okunur: her (ses, ton) için bir kez ölçülür (6 cümle), sonra diskten."""
    global _SES, _HIZ_DOSYASI
    t0 = time.perf_counter()
    s = Seslen.__new__(Seslen)              # Seslen.__init__ onnxruntime ve özdenetim arar; burada gerekmiyor
    s.model = _AndroidModeli(klasor, kopru, is_parcacigi)
    s._duygu, s._hiz, s._oz = {}, {}, None
    _HIZ_DOSYASI = Path(klasor) / "dogal_hiz.json"
    try:
        s._hiz = {tuple(k.split("/")): v for k, v in json.loads(_HIZ_DOSYASI.read_text()).items()}
    except (OSError, ValueError):
        pass
    s.model.okunus("Merhaba.")
    _SES = s
    return time.perf_counter() - t0


def ayarla(tasiyici_kat: float) -> None:
    global TASIYICI_KAT
    TASIYICI_KAT = float(tasiyici_kat)


def isit(ses: str, ton: str) -> float:
    """İlk istekten önce: sesin doğal hızı (yoksa ölç, diske yaz), harf adları, bir kısa üretim (ONNX Runtime'ın
    ilk çalıştırması yavaş). Süre (sn)."""
    t0 = time.perf_counter()
    ton = ton if ton in _SES.tonlar(ses) else "notr"
    if (ses, ton) not in _SES._hiz:
        _SES.dogal_hiz(ses, ton)
        try:
            _HIZ_DOSYASI.write_text(json.dumps({f"{a}/{b}": v for (a, b), v in _SES._hiz.items()}))
        except OSError:
            pass
    _SES.model._harf_adlari()
    _SES.model.konus("Tamam.", _SES.model.stil(ses, ton))
    return time.perf_counter() - t0


def seslendir(metin: str, ses: str, ton: str = "otomatik", hiz: float = 1.0, perde: float = 1.0):
    """Her cümle: (cümle, float32 dalga baytları, sonraki sessizlik ms, ton, üretim ms, aşamalar)."""
    _SURE.clear()
    t0 = time.perf_counter()
    for p in _SES.seslendir(metin, ses=ses, ton=ton, hiz=hiz if hiz != 1.0 else None, perde=perde):
        sure = (time.perf_counter() - t0) * 1000
        asama = f"okunuş {_SURE.get('okunus', 0):.0f}, metin {_SURE.get('metin', 0):.0f}, ses {_SURE.get('ses', 0):.0f} ms"
        yield p.cumle, p.ses.astype(np.float32).tobytes(), int(p.sessizlik_ms), p.ton, sure, asama
        _SURE.clear()
        t0 = time.perf_counter()
