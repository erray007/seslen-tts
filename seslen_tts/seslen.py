# -*- coding: utf-8 -*-
"""
SESLEN — metni cümle cümle, duygusu ve üslubuyla seslendirir.

  ses      16 ses (Seslen.sesler): nehir, ege, ayse, serdar, zeynep, emre, selin, fatma, hasan, elif, can, ece,
           pitircik, koca, firfir, kara
  ton      "otomatik" (her cümleye turkce-duygu karar verir) ya da notr, neseli, heyecan, uyari, merak, ozur,
           uzgun, kizgin, fisilti (bütün metin o tonda)
  uslup    "otomatik" (turkce-uslup metnin türünü bulur: haber, masal, talimat…) ya da üslup adı; hız, ezgi
           genişliği ve duraklamaları belirler. hiz verilirse üslubun hızını ezer.
turkce-duygu ve turkce-uslup isteğe bağlıdır (pip install "seslen-tts[hepsi]"); yoksa her cümle nötr, üslup düz.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from .model import ORAN, SesModeli

TONLAR = ("notr", "neseli", "heyecan", "uyari", "merak", "ozur", "uzgun", "kizgin", "fisilti")


@dataclass
class Parca:
    cumle: str
    ses: np.ndarray              # 24 kHz, float32
    sessizlik_ms: int            # bu cümleden sonraki duraklama
    ton: str
    bilgi: dict = field(default_factory=dict)


# Tek başına okunan 1-2 sözcüklük cümleye model sonda uydurma hece ekliyor ("Dikkat!" → "dikkatsınız"); sonraki
# cümleye virgülle bağlanınca temiz okunuyor ve turkce-duygu tonu da doğru buluyor ("Dikkat, bu işlem…" → uyarı).
_KISA = re.compile(r"(?:^|(?<=[.!?…]\s))((?:[^\s.!?…;]+\s)?([^\s.!?…;]+))([.!]+)\s+(\w)(?=(\w?))")
_KISALTMA = {"prof", "doç", "dr", "sn", "av", "bkz", "vb", "vs", "örn", "yrd", "mah", "cad", "sok", "no", "op", "uzm"}


def _kucuk(s: str) -> str:
    return s.replace("I", "ı").replace("İ", "i").lower()


def akici(metin: str) -> str:
    """Kısa ünlem ya da cümleyi sonrakine virgülle bağlar: "Merhaba! Ben Ayşe." → "Merhaba, ben Ayşe." """
    def bagla(m: re.Match) -> str:
        son, isaret, ilk, ikinci = m.group(2), m.group(3), m.group(4), m.group(5)
        if isaret == "." and (len(son) < 4 or _kucuk(son) in _KISALTMA or any(c.isdigit() for c in m.group(1))):
            return m.group(0)                  # "Dr. Ahmet", "Madde 3. Sonra" bağlanmaz
        if not ikinci or ikinci.islower():       # "ABD" gibi kısaltmanın büyük harfi kalır
            ilk = _kucuk(ilk)
        return f"{m.group(1)}, {ilk}"
    for _ in range(3):                         # "Evet! Tamam! Gidelim." birden çok kısa cümle
        metin, eski = _KISA.sub(bagla, metin), metin
        if metin == eski:
            break
    return metin


def _cumleler(metin: str) -> list[str]:
    metin = akici(" ".join(metin.split()))
    try:
        from turkce_duygu import cumlelere_bol
        return cumlelere_bol(metin)
    except ImportError:
        return [c for c in re.split(r"(?<=[.!?…])\s+", " ".join(metin.split())) if c]


class Seslen:
    def __init__(self, klasor=None, is_parcacigi: int = 0):
        self.model = SesModeli(klasor, is_parcacigi)
        self._duygu: dict = {}

    @property
    def sesler(self) -> dict:
        return self.model.sesler

    def tonlar(self, ses: str) -> tuple[str, ...]:
        return tuple(self.model.sesler[ses]["tonlar"])

    def _okuma(self, metin: str, uslup: str, hiz: float | None):
        varsayilan = {"hiz": 1.0, "tonlama": 1.0, "duraklama_ms": 350, "virgul_ms": 0}
        try:
            from turkce_uslup import okuma_ayari
            from turkce_uslup import uslup as uslup_bul
        except ImportError:
            return "duz", {**varsayilan, **({"hiz": hiz} if hiz else {})}
        ad = uslup_bul(metin[:4000]).uslup if uslup == "otomatik" else uslup
        a = okuma_ayari(ad, **({"hiz": hiz} if hiz else {}))
        return ad, {"hiz": a.hiz, "tonlama": a.tonlama, "duraklama_ms": a.duraklama_ms, "virgul_ms": a.virgul_ms}

    def _tonlar(self, ses: str, cumleler: list[str]) -> list[tuple[str, float]]:
        """Her cümle için (ton, yoğunluk 0-1); turkce-duygu yoksa nötr."""
        try:
            from turkce_duygu import Cozumleyici
        except ImportError:
            return [("notr", 0.0)] * len(cumleler)
        if ses not in self._duygu:
            self._duygu[ses] = Cozumleyici(stiller=self.tonlar(ses))
        sonuc = self._duygu[ses].metin_cozumle(" ".join(cumleler))
        if len(sonuc) != len(cumleler):           # bölücüler ayrıştıysa cümle cümle
            sonuc = [self._duygu[ses].metin_cozumle(c)[0] for c in cumleler]
        return [(r.stil, float(r.yogunluk)) for r in sonuc]

    def seslendir(self, metin: str, ses: str = "zeynep", ton: str = "otomatik", yogunluk: float | None = None,
                  hiz: float | None = None, perde: float = 1.0, uslup: str = "otomatik") -> Iterator[Parca]:
        """Cümle cümle Parca üretir: ilk cümle hazır olur olmaz çalınabilir."""
        if ses not in self.model.sesler:
            raise ValueError(f"bilinmeyen ses: {ses}; sesler: {', '.join(self.model.sesler)}")
        if ton != "otomatik" and ton not in self.tonlar(ses):
            raise ValueError(f"{ses} için ton yok: {ton}; tonlar: {', '.join(self.tonlar(ses))}")
        uslup_ad, ok = self._okuma(metin, uslup, hiz)
        cumleler = _cumleler(metin)
        secim = self._tonlar(ses, cumleler) if ton == "otomatik" else [(ton, 1.0)] * len(cumleler)
        notr = self.model.stil(ses, "notr")
        for c, (t, y) in zip(cumleler, secim):
            t = t if t in self.tonlar(ses) else "notr"
            s = self.model.stil(ses, t)
            if t != "notr":           # yoğunluk: nötrden tona doğru; otomatikte hafif duygu da duyulsun diye 0,5 +
                k = yogunluk if yogunluk is not None else (0.5 + y if ton == "otomatik" else 1.0)
                s = notr + k * (s - notr)
            w = self.model.konus(c, s, hiz=ok["hiz"], perde=perde, tonlama=ok["tonlama"], virgul_ms=ok["virgul_ms"])
            yield Parca(c, w, ok["duraklama_ms"], t, {"uslup": uslup_ad, "yogunluk": round(y, 2)})

    def tum(self, metin: str, **ayar) -> np.ndarray:
        """Bütün metin tek dalga (cümle aralarında üslubun duraklaması)."""
        parcalar = list(self.seslendir(metin, **ayar))
        dalga = []
        for i, p in enumerate(parcalar):
            dalga.append(p.ses)
            if i < len(parcalar) - 1:
                dalga.append(np.zeros(int(ORAN * p.sessizlik_ms / 1000), np.float32))
        return np.concatenate(dalga) if dalga else np.zeros(0, np.float32)


def wav_yaz(yol, dalga: np.ndarray, oran: int = ORAN) -> None:
    """16 bit tek kanal WAV (yalnız standart kütüphane)."""
    import wave
    x = np.clip(dalga / max(1e-6, float(np.abs(dalga).max())) * 0.9, -1, 1)
    with wave.open(str(yol), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(oran)
        w.writeframes((x * 32767).astype("<i2").tobytes())
