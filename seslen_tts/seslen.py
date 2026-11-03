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
#: Bu üsluplarda (turkce-uslup) metin belge gibi okunur: duygu sabit ve hafif (bkz. _belge_tonu).
BELGE_USLUPLARI = {"haber", "bilgi", "deneme", "talimat"}
BELGEDE_YOK = {"heyecan", "kizgin", "uzgun", "fisilti"}
BELGE_ESIK = 0.5
BELGE_YOGUNLUK = 0.6            # belgede tonun en çok bu kadarı (nötrden tona doğru)
UZUN_CUMLE = 220                # bundan uzun cümle bölünür (karakter; PL-BERT 512 işaret sınırının çok altında)
PARAGRAF_KAT = 2.0              # paragraf sonunda duraklama üslubun duraklamasının bu katı
# UZUN CÜMLE: bundan çok sözcüklü cümle virgül / noktalı virgül / iki noktadan parçalara bölünüp ayrı okunur, parça
# arasında PARCA_MS durulur. Genel metinden 40 uzun cümlede (Seslen araclar/deney/bolme.py) takılan 18 → 14, UTMOS
# 3,05 → 3,09: model uzun cümlede sesleri kaydırıyor.
PARCA_SOZCUK = 12
PARCA_MS = 175                  # parça sonundaki virgül dalgada kesilir (model.SON_PAY): virgül + 100 ms kadar
CUMLE_MS = 450                  # cümle sonunda en az (haber üslubu 350 ms idi; kulak: "noktadan sonra bekleme yok";
                                # 03.11'den beri noktanın kendi süresi kesildiği için 400 → 450)
BELGE_KISA = 4                  # belgede bundan az sözcüklü cümle (başlık, madde) nötr okunur
#: Sesin doğal hızını ölçen orta uzunlukta, düz cümleler (bkz. Seslen.dogal_hiz).
AYAR_CUMLELERI = (
    "Bugün hava oldukça güzel ve güneşli görünüyor.",
    "Toplantı yarın sabah erken saatte başlayacak.",
    "Kitabı okudum ama sonunu pek beğenmedim.",
    "Yeni proje için ekip arkadaşlarıyla görüştük.",
    "Şehrin merkezinde küçük bir kafe açılmış.",
    "Raporun son bölümünü akşama kadar bitireceğim.",
)


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


def _bol(metin: str) -> list[str]:
    metin = akici(" ".join(metin.split()))
    try:
        from turkce_duygu import cumlelere_bol
        return cumlelere_bol(metin)
    except ImportError:
        return [c for c in re.split(r"(?<=[.!?…])\s+", metin) if c]


def _bolumler(metin: str) -> list[tuple[str, bool]]:
    """Belge (Markdown / HTML / düz metin) → [(cümle, paragrafın son cümlesi mi)]. Belge okuyucu
    (turkce_normalize.paragraflar) başlık, liste, tablo ve biçim işaretlerini cümle yapısına çevirir; modelin işaret
    sınırını aşan cümle bölünür (uzun_cumleyi_bol)."""
    try:
        from turkce_normalize import paragraflar, uzun_cumleyi_bol
    except ImportError:                                   # pragma: no cover — turkce-vurgu[tam] onu kurar
        paragraflar = lambda m: [p for p in re.split(r"\n\s*\n", m) if p.strip()]       # noqa: E731
        uzun_cumleyi_bol = lambda c, en_cok=220: [c]                                       # noqa: E731
    cikti = []
    for p in paragraflar(metin):
        cumleler = [q for c in _bol(p) for q in uzun_cumleyi_bol(c, UZUN_CUMLE)]
        cikti += [(c, i == len(cumleler) - 1) for i, c in enumerate(cumleler)]
    return cikti


def _cumleler(metin: str) -> list[str]:
    return [c for c, _ in _bolumler(metin)]


def _belge_tonu(ton: str, yogunluk: float) -> tuple[str, float]:
    """Makale, haber, bilgi metni, talimat: duygu sabit ve hafif. Güçlü duygular (heyecan, kızgın, üzgün, fısıltı)
    hiç seçilmez; ötekiler yalnız belirginse (yoğunluk ≥ BELGE_ESIK), o da hafif. 03.11.2026: bir makalede cümle
    cümle heyecan → kızgın → merak → uyarı geçiyordu ("izin gerekmez" kızgın okundu)."""
    if ton in BELGEDE_YOK or yogunluk < BELGE_ESIK:
        return "notr", 0.0
    return ton, yogunluk


def _virgulden_bol(cumle: str) -> list[str]:
    try:
        from turkce_normalize import virgulden_bol
    except ImportError:                                   # pragma: no cover — eski turkce-normalize
        return [cumle]
    return virgulden_bol(cumle, PARCA_SOZCUK)


class Seslen:
    def __init__(self, klasor=None, is_parcacigi: int = 0, denetim: bool | None = None):
        """denetim: özdenetim (her cümle dinlenir, bozuk sözcük onarılır; seslen_tts.denetim). None: tanıyıcı
        (denetim.onnx) model klasöründe varsa açık; True: yoksa Hugging Face'ten iner (~360 MB); False: kapalı."""
        from .denetim import Ozdenetim, Taniyici
        from .model import denetim_klasoru
        self.model = SesModeli(klasor, is_parcacigi)
        self._duygu: dict = {}
        self._hiz: dict = {}
        self._oz = None
        if denetim is not False:
            k = denetim_klasoru(self.model.klasor, indir=bool(denetim))
            if k is not None:
                self._oz = Ozdenetim(self.model, Taniyici(k, is_parcacigi))
            elif denetim:
                raise FileNotFoundError("özdenetim tanıyıcısı (denetim.onnx) bulunamadı ve indirilemedi")

    @property
    def denetim(self) -> bool:
        return self._oz is not None

    @property
    def sesler(self) -> dict:
        return self.model.sesler

    def tonlar(self, ses: str) -> tuple[str, ...]:
        return tuple(self.model.sesler[ses]["tonlar"])

    def dogal_hiz(self, ses: str, ton: str, stil: np.ndarray | None = None) -> float:
        """Sesin bu tondaki doğal hızı (hece/sn): AYAR_CUMLELERI'nin ortancası, ses × ton başına bir kez ölçülür.
        Model uzun cümleyi bundan hızlı okuyor (21 sözcükte 7,9 hece/sn, kısa cümlede 5,3); seslendir her cümleyi
        bunun bandına çeker (model.HIZ_BANDI)."""
        if (ses, ton) not in self._hiz:
            s = self.model.stil(ses, ton) if stil is None else stil
            self._hiz[(ses, ton)] = float(np.median([self.model.hece_hizi(c, s) for c in AYAR_CUMLELERI]))
        return self._hiz[(ses, ton)]

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
        bolumler = _bolumler(metin)
        cumleler = [c for c, _ in bolumler]
        uslup_ad, ok = self._okuma(" ".join(cumleler), uslup, hiz)
        belge = uslup_ad in BELGE_USLUPLARI and ton == "otomatik"
        secim = self._tonlar(ses, cumleler) if ton == "otomatik" else [(ton, 1.0)] * len(cumleler)
        if belge:                     # kısa cümle (başlık, madde) belgede hep nötr: "İki set." neşeli okunuyordu
            secim = [_belge_tonu(t, y) if len(c.split()) >= BELGE_KISA else ("notr", 0.0)
                     for c, (t, y) in zip(cumleler, secim)]
        notr = self.model.stil(ses, "notr")
        stiller = []
        for t, y in secim:
            t = t if t in self.tonlar(ses) else "notr"
            s = self.model.stil(ses, t)
            if t != "notr":           # yoğunluk: nötrden tona doğru; otomatikte hafif duygu da duyulsun diye 0,5 +
                # 1'i aşmaz (0.3'e kadar 0,5 + y idi, 1,5'e çıkıp tonu abartıyordu); belgede en çok BELGE_YOGUNLUK
                k = yogunluk if yogunluk is not None else (min(1.0, 0.5 + y) if ton == "otomatik" else 1.0)
                if belge and yogunluk is None:
                    k = min(k, BELGE_YOGUNLUK)
                s = notr + k * (s - notr)
            stiller.append((t, s))
        for (c, son), (t, s), (_, y) in zip(bolumler, stiller, secim):
            ayar = {"perde": perde, "tonlama": ok["tonlama"], "virgul_ms": ok["virgul_ms"],
                    "hedef_hece": self.dogal_hiz(ses, t) * ok["hiz"]}
            parcalar = _virgulden_bol(c)                 # uzun cümle virgülden: model uzunda sesleri kaydırıyor
            for j, p in enumerate(parcalar):
                bilgi = {"uslup": uslup_ad, "yogunluk": round(y, 2), "belge": belge}
                if self._oz is not None:
                    w, bilgi["denetim"] = self._oz.uret(p, s, notr=notr, hiz=ok["hiz"], **ayar)
                else:
                    w = self.model.konus(p, s, hiz=ok["hiz"], **ayar)
                if j < len(parcalar) - 1:
                    sessizlik = PARCA_MS                 # cümle içinde: virgül kadar
                else:
                    sessizlik = int(max(ok["duraklama_ms"], CUMLE_MS) * (PARAGRAF_KAT if son else 1))
                yield Parca(p, w, sessizlik, t, bilgi)

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
