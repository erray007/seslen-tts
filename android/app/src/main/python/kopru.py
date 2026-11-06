# -*- coding: utf-8 -*-
"""
KÖPRÜ — seslen_tts'i Android'de çalıştırır. Bütün mantık (okunuş, hizalama, soru ezgisi, duraklamalar, kısa cümle
taşıyıcısı, hız dengesi) seslen_tts'in kendisi; yalnız ONNX oturumları Android'in ONNX Runtime'ına (OrtKopru.kt)
yönlenir: Python'un onnxruntime paketi Android'de yok.

Kotlin şunları çağırır: hazirla(klasor, kopru, is_parcacigi), sesler(), seslendir(metin, ses, ton, hiz) → yineleyici
(her adımda (cümle, float32 dalga baytları, sessizlik ms, ton, süre ms)).
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
        sonuc = self.kopru.calistir(self.ad, adlar, sekiller, turler, veriler)
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


def hazirla(klasor: str, kopru, is_parcacigi: int = 4) -> float:
    """Modeli açar, okunuş çeviricisini ısıtır; süre (sn)."""
    global _SES
    t0 = time.perf_counter()
    s = Seslen.__new__(Seslen)              # Seslen.__init__ onnxruntime ve özdenetim arar; burada gerekmiyor
    s.model = _AndroidModeli(klasor, kopru, is_parcacigi)
    s._duygu, s._hiz, s._oz = {}, {}, None
    s.model.okunus("Merhaba.")
    _SES = s
    return time.perf_counter() - t0


def sesler() -> str:
    """JSON: [[ad, görünen ad, açıklama], …]"""
    return json.dumps([[a, b["gorunen"], b.get("aciklama", "")] for a, b in _SES.sesler.items()], ensure_ascii=False)


def seslendir(metin: str, ses: str, ton: str = "otomatik", hiz: float = 1.0):
    t0 = time.perf_counter()
    for p in _SES.seslendir(metin, ses=ses, ton=ton, hiz=hiz if hiz != 1.0 else None):
        sure = (time.perf_counter() - t0) * 1000
        yield p.cumle, p.ses.astype(np.float32).tobytes(), int(p.sessizlik_ms), p.ton, sure
        t0 = time.perf_counter()
