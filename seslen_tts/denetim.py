# -*- coding: utf-8 -*-
"""
ÖZDENETİM — her cümle üretilince dinlenir; bir sözcük bozuk okunduysa onarım adayları denenir, en az kusurlu olan
seçilir. PyTorch gerekmez: Türkçe CTC tanıyıcı (wav2vec2 XLS-R 300M, Common Voice 8 Türkçe;
mpoyraz/wav2vec2-xls-r-300m-cv8-turkish, Apache-2.0) ONNX'e aktarılıp 8 bite indirildi (denetim.onnx, ~360 MB).

Neden (03.11.2026): model uzun ve seyrek sözcüklü cümlede sesleri kaydırıyor ("soru" → "sonru", "kanun" → "kanın").
Genel metinden 40 uzun cümlede (Seslen araclar/deney/bolme.py) takılan 18; bu yolun Seslen'deki eşiyle çok azalıyor.

Bozuk sözcük: telaffuz puanı (GOP: metin CTC ile zorunlu hizalanır, harfin karelerinde log P(harf) − log P(en olası
harf)) ESIK_GOP'un altında, ya da şüpheli (< ESIK_SUPHE) ve tanıyıcı o sözcükte başka harf yazdı (ğ = y sayılır).
Onarımlar, sırayla (ilk kusursuzda durulur): bozuk sözcüğün süresi ×1,3 / ×1,6, yanlış duyulan ünlüsü ×2, vurgu
işareti kalkar, öncesine virgül, cümle nötr tonda, bütün cümle %10 yavaş. Kusursuz cümlede bedel bir tanıyıcı geçişi
(işlemcide 5 sn ses ~0,5 sn), kusurluda aday başına bir üretim + bir geçiş.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

DOSYALAR = ["denetim.onnx", "denetim.json"]
ESIK_GOP = -5.0
ESIK_SUPHE = -0.8      # bu puanın altında tanıyıcı başka harf de duyduysa bozuk ("rögarları" → "rogarları" −0,94; 03.11)
YAVAS = (1.3, 1.6)
UNLU_KAT = 2.0
# STİL SAPMASI: süre ve vurgu onarımları ünlü kaymasını ("giriş" → "geriş", "rögar" → "rogar") düzeltemiyordu; stil
# vektörüne küçük, tohumu sabit bir sapma (SAPMA × stilin std'si) modeli o sözcüğü yeniden "zar atarak" söyletir.
# 3 ses × 6 inatçı cümlede birebir duyulan: ilk okuma 9/18, sapmalı her okuma ~%50; σ 0,1 0,2-0,3 kadar etkili,
# sesi en az değiştiren (03.11.2026). Tohumlar SAPMA_TOHUM; "sınıf", "EVREN'de" hiçbir tohumla düzelmedi (model sınırı).
SAPMA, SAPMA_TOHUM = 0.1, (0, 1)
_HARF = re.compile(r"[^a-zçğıöşüâîû ]")
_UNLU = set("aeıioöuü")
_IPA_UNLU = set("aeɯioøuyœæɛ")


def normal(t: str) -> str:
    t = t.replace("I", "ı").replace("İ", "i").lower().replace("â", "a").replace("î", "i").replace("û", "u")
    return " ".join(_HARF.sub(" ", t.replace("'", "").replace("’", "")).split())


def yeniden_ornekle(x: np.ndarray, oran: int, hedef: int = 16000) -> np.ndarray:
    """Çok fazlı yeniden örnekleme (scipy.signal.resample_poly'nin eşi, scipy'siz): yukarı ×up, Kaiser pencereli
    sinc alçak geçiren, aşağı ÷down. Doğrusal ara değerleme yüksek frekansı bozuyor, tanıyıcının puanı düşüyordu."""
    from math import gcd
    if oran == hedef:
        return np.asarray(x, np.float32)
    g = gcd(oran, hedef)
    up, down = hedef // g, oran // g
    kesim = 1.0 / max(up, down)                          # Nyquist'e göre
    yari = 10 * max(up, down)
    n = np.arange(-yari, yari + 1)
    h = kesim * np.sinc(kesim * n) * np.kaiser(len(n), 5.0) * up
    u = np.zeros(len(x) * up, np.float64)
    u[::up] = x
    y = np.convolve(u, h)[yari::down]
    return y[: int(np.ceil(len(x) * up / down))].astype(np.float32)


def hizala(ref: str, hyp: str) -> list[tuple[str | None, str | None]]:
    """Harf düzeyinde Levenshtein hizası (boşluklar yok sayılır)."""
    r, h = ref.replace(" ", ""), hyp.replace(" ", "")
    n, m = len(r), len(h)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0], d[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    yol, i, j = [], n, m
    while i or j:
        if i and j and d[i, j] == d[i - 1, j - 1] + (r[i - 1] != h[j - 1]):
            yol.append((r[i - 1], h[j - 1])); i -= 1; j -= 1
        elif i and d[i, j] == d[i - 1, j] + 1:
            yol.append((r[i - 1], None)); i -= 1
        else:
            yol.append((None, h[j - 1])); j -= 1
    return yol[::-1]


def _denk(s: str) -> str:
    return s.replace("ğ", "y")


class Taniyici:
    def __init__(self, klasor: Path, is_parcacigi: int = 0):
        import os
        import onnxruntime as ort
        se = ort.SessionOptions()
        se.intra_op_num_threads = is_parcacigi or min(6, os.cpu_count() or 1)
        self.oturum = ort.InferenceSession(str(klasor / "denetim.onnx"), se, providers=["CPUExecutionProvider"])
        b = json.loads((klasor / "denetim.json").read_text(encoding="utf-8"))
        self.sozluk, self.bosluk, self.dolgu = b["sozluk"], b["bosluk"], b["dolgu"]
        self.ters = {v: k for k, v in self.sozluk.items()}
        self.olcekle = b.get("olcekle", True)

    def logp(self, x: np.ndarray, oran: int) -> np.ndarray:
        y = yeniden_ornekle(np.asarray(x, np.float32).reshape(-1), oran)
        if self.olcekle:
            y = (y - y.mean()) / np.sqrt(y.var() + 1e-7)
        g = self.oturum.run(None, {"ses": y[None].astype(np.float32)})[0][0]
        e = g.max(-1, keepdims=True)
        return g - e - np.log(np.exp(g - e).sum(-1, keepdims=True))

    def duy(self, lp: np.ndarray) -> str:
        yol, onceki = [], None
        for t in lp.argmax(-1):
            if t != onceki and t != self.dolgu:
                yol.append(" " if t == self.bosluk else self.ters.get(int(t), ""))
            onceki = t
        return normal("".join(yol))

    def gop(self, lp: np.ndarray, ref: str) -> list[tuple[str, float]]:
        """Sözcük başına en kötü harfin puanı (CTC Viterbi zorunlu hizalaması; Seslen denetim.gop ile aynı)."""
        hedef = [self.bosluk if c == " " else self.sozluk.get(c, self.bosluk) for c in ref]
        S = [self.dolgu]
        for t in hedef:
            S += [t, self.dolgu]
        T, L = lp.shape[0], len(S)
        if T < L // 2 + 1:
            return [(w, -20.0) for w in ref.split()]
        e = lp[:, S]
        d = np.full((T, L), -1e9)
        bp = np.zeros((T, L), dtype=np.int32)
        d[0, :2] = e[0, :2]
        atla = np.array([s >= 2 and S[s] != self.dolgu and S[s] != S[s - 2] for s in range(L)])
        for t in range(1, T):
            a = d[t - 1]
            yig = np.stack([a, np.concatenate([[-1e9], a[:-1]]), np.where(atla, np.concatenate([[-1e9, -1e9], a[:-2]]), -1e9)])
            k = yig.argmax(0)
            d[t] = yig[k, np.arange(L)] + e[t]
            bp[t] = np.arange(L) - k
        s = L - 1 if d[T - 1, L - 1] > d[T - 1, L - 2] else L - 2
        en_iyi = lp.max(1)
        harf: dict[int, list[float]] = {}
        for t in range(T - 1, -1, -1):
            if s % 2 == 1:
                harf.setdefault(s // 2, []).append(lp[t, S[s]] - en_iyi[t])
            s = bp[t, s]
        puan = [float(np.mean(harf[i])) if i in harf else -10.0 for i in range(len(hedef))]
        sonuc, i = [], 0
        for w in ref.split():
            sonuc.append((w, round(min(puan[i:i + len(w)]), 2)))
            i += len(w) + 1
        return sonuc


def harf_hatasi(ref: str, duyulan: str) -> list[int]:
    sozcukler = ref.split()
    sahip = [i for i, s in enumerate(sozcukler) for _ in s]
    hata, ri = [0] * len(sozcukler), 0
    for a, b in hizala(_denk(ref), _denk(duyulan)):
        if a is None:
            if 0 < ri < len(sahip):
                hata[sahip[ri - 1]] += 1
            continue
        hata[sahip[ri]] += b is None or a != b
        ri += 1
    return hata


def yanlis_unluler(ref: str, duyulan: str) -> list[set[int]]:
    sozcukler = ref.split()
    sonuc: list[set[int]] = [set() for _ in sozcukler]
    harf = [(i, c) for i, s in enumerate(sozcukler) for c in s]
    sira, onceki, k = [], -1, 0
    for i, c in harf:
        k = 0 if i != onceki else k
        sira.append(k if c in _UNLU else None)
        k += c in _UNLU
        onceki = i
    ri = 0
    for a, b in hizala(_denk(ref), _denk(duyulan)):
        if a is None:
            continue
        if (b is None or a != b) and sira[ri] is not None:
            sonuc[harf[ri][0]].add(sira[ri])
        ri += 1
    return sonuc


class Ozdenetim:
    def __init__(self, model, taniyici: Taniyici):
        self.model, self.t = model, taniyici

    def dinle(self, w: np.ndarray, oran: int, ref: str) -> tuple[float, list[int], str]:
        """→ (ceza, bozuk sözcüklerin sırası, duyulan). Ceza 0 kusursuz."""
        lp = self.t.logp(w, oran)
        g, duyulan = self.t.gop(lp, ref), self.t.duy(lp)
        ceza, bozuk = 0.0, []
        for i, ((s, v), h) in enumerate(zip(g, harf_hatasi(ref, duyulan))):
            supheli = v < ESIK_SUPHE and h > 0 and len(s) >= 3
            if v < ESIK_GOP or supheli:
                bozuk.append(i)
            ceza += max(0.0, ESIK_GOP - v) + (h if supheli else 0)
        return round(ceza, 2), bozuk, duyulan

    def _onar(self, okunus: str, degisim: dict[int, str]):
        """Sözcük onarımları ({sözcük sırası: onarım}) → (okunus, uzat) ya da None."""
        m = self.model
        parca = okunus.split(" ")
        sozcuk = [j for j, p in enumerate(parca) if any(c.isalpha() for c in p)]
        uygulanan = {}
        for i, o in degisim.items():
            if i >= len(sozcuk):
                continue
            j = sozcuk[i]
            if o == "vurgusuz":
                if "ˈ" not in parca[j]:
                    continue
                parca[j] = parca[j].replace("ˈ", "")
            elif o == "durak":
                if i == 0 or not parca[sozcuk[i - 1]][-1:].isalpha():
                    continue
                parca[sozcuk[i - 1]] += ","
            uygulanan[i] = o
        if not uygulanan:
            return None
        okunus = " ".join(parca)
        araliklar = m.sozcuk_araliklari(okunus)
        if len(araliklar) != len(sozcuk):
            return None
        uzat = []
        for i, o in uygulanan.items():
            bas, son = araliklar[i]
            if o.startswith("yavas"):
                uzat.append((bas, son, YAVAS[o == "yavas2"]))
            elif o.startswith("unlu:"):
                kod = [c for c in okunus.split(" ")[sozcuk[i]] if c in m.semboller]
                unluler = [bas + k for k, c in enumerate(kod) if c in _IPA_UNLU]
                uzat += [(unluler[k], unluler[k] + 1, UNLU_KAT) for k in map(int, o[5:].split(",")) if k < len(unluler)]
        return okunus, uzat or None

    def uret(self, metin: str, stil: np.ndarray, notr: np.ndarray | None = None, hiz: float = 1.0, **kw):
        """Cümleyi üretir, dinler, gerekirse onarır → (dalga, bilgi)."""
        from turkce_vurgu import normallestir
        from .model import ORAN
        m = self.model
        ref = normal(normallestir(metin))
        w = m.konus(metin, stil, hiz=hiz, **kw)
        if not ref:
            return w, {"ceza": 0.0}
        ceza, bozuk, duyulan = self.dinle(w, ORAN, ref)
        bilgi = {"ceza": ceza, "duyulan": duyulan, "onarim": None}
        if ceza == 0 or len(metin.split()) <= 2:          # kısa cümle taşıyıcıyla üretilir; onarım okunuşu bozar
            return w, bilgi
        okunus = m.okunus(metin)
        eslesir = len(m.sozcuk_araliklari(okunus)) == len(ref.split())
        bozuk = bozuk if eslesir else []
        unlu = yanlis_unluler(ref, duyulan)
        en_iyi = (ceza, w, None, duyulan)
        adaylar = [(o, {i: o for i in bozuk}) for o in ("yavas", "yavas2")]
        adaylar.append(("unlu", {i: "unlu:" + ",".join(map(str, sorted(unlu[i]))) for i in bozuk if unlu[i]}))
        adaylar += [(o, {i: o for i in bozuk}) for o in ("vurgusuz", "durak")]
        adaylar += [("notr", None), ("hiz", None)] + [(f"sapma{t}", None) for t in SAPMA_TOHUM]
        for ad, degisim in adaylar:
            if ad == "notr":
                if notr is None or np.allclose(stil, notr):
                    continue
                w2 = m.konus(metin, notr, hiz=hiz, **kw)
            elif ad.startswith("sapma"):
                s = np.asarray(stil, np.float32)
                r = np.random.default_rng(int(ad[5:]))
                w2 = m.konus(metin, s + SAPMA * s.std() * r.standard_normal(s.shape).astype(np.float32), hiz=hiz, **kw)
            elif ad == "hiz":
                w2 = m.konus(metin, stil, hiz=hiz * 0.9, **kw)
            else:
                a = self._onar(okunus, degisim) if degisim else None
                if a is None:
                    continue
                w2 = m.konus(metin, stil, hiz=hiz, okunus=a[0], uzat=a[1], **kw)
            c2, _, d2 = self.dinle(w2, ORAN, ref)
            if c2 < en_iyi[0]:
                en_iyi = (c2, w2, ad, d2)
            if c2 == 0:
                break
        c, w, ad, d = en_iyi
        bilgi.update({"ceza": c, "ilk_ceza": ceza, "onarim": ad, "duyulan": d})
        return w, bilgi
