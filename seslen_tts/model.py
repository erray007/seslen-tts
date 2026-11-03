# -*- coding: utf-8 -*-
"""
SES MODELİ — Türkçe okunuştan sese, yalnız ONNX Runtime ile (PyTorch gerekmez), işlemcide.

İki ağ (Seslen'in StyleTTS2 öğrencisinden aktarıldı):
  metin.onnx  işaretler + stil → süre kodlayıcı çıkışı (d), metin kodlayıcı çıkışı (t_en), işaret başına süre
  ses.onnx    hizalanmış özellikler + stil + perde/tonlama + soru ezgisi → 24 kHz dalga
Aradaki hizalama (süreleri yuvarlama, çerçevelere yayma) burada numpy ile yapılır.

Model klasörü (öncelik sırasıyla): SesModeli(klasor), SESLEN_TTS_MODEL ortam değişkeni, Hugging Face
(SESLEN_TTS_HF deposu, ilk kullanımda iner ve önbellekte kalır; SESLEN_TTS_INDIRME=0 ile kapalı).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

ORAN = 24000
HF_DEPO = os.environ.get("SESLEN_TTS_HF", "seslen-tr/seslen-tts")
DOSYALAR = ["metin.onnx", "ses.onnx", "stiller.npz", "sesler.json", "semboller.json"]
BAS_DOLGU = 4          # baştaki dolgu işaretinin en uzun süresi (25 ms birim); uzarsa model boşluğu sesle dolduruyor
BIRIM_MS = 25          # bir süre birimi = 2 mel çerçevesi = 600 örnek
ORNEK_BIRIM = 600
NOKTALI_MS = 200       # noktalı virgülde en az duraklama (yoksa virgülden ayırt edilmiyor)
# KISA CÜMLE: tek başına 1-2 sözcüklük cümleye model sonda uydurma hece ekliyor ("Evet." → "evetnı"). Arkasına taşıyıcı
# cümle eklenip üretilir, ses hizalamadan bilinen son harfte kesilir (16 ses × 19 kısa cümlede takılan 51 → 8).
# Taşıyıcı "h" ile başlar: "Bunu…" ile başlayınca dudak sesi kesimden önce sızıyordu ("Katkı." → "katkım", "Bitti." →
# "bittim"; 4 ses × 20 kısa cümlede birebir duyulan 49 → 71, 03.11.2026).
KISA_SOZCUK = 2
TASIYICI = " Hem bunu sana daha önce de söylemiştim."
# SORU EZGİSİ: evet/hayır sorusunda ezginin tepesi soru ekinden (mı/mi/mu/mü + kişi ekleri) önceki sözcüğün vurgulu
# hecesindedir ("yardımcı olaBİLİR misin?", "ANkara mı?"; Göksel & Kerslake §21.2). Model bunu düz geçtiği için o
# heceye bu kadar yarım ton eklenir (ses.onnx içinde ~175 ms ile yumuşatılır). Soru sözcüklü sorulara dokunmaz.
SORU_TEPE = 6.0
# HIZ DENGESİ: model uzun cümleyi hızlı, kısa cümleyi yavaş okuyor (bir makalede 4,6–7,3 hece/sn, 03.11.2026).
# Seslen.seslendir belgenin ortanca hızını bulur; her cümle bunun ±HIZ_BANDI içine çekilir (en çok HIZ_EN_COK kat).
# YEREL HIZ ve DURAKLAMA (03.11.2026, kulak: "bazı yerlerde cümleyi çok hızlı okumaya başlıyor", "virgül ve noktaya
# bazen uyuyor bazen uymuyor"): cümle içinde YEREL_HECE heceli bölüm cümle ortalamasının YEREL_KAT katından hızlıysa
# yavaşlatılır; virgülde en az VIRGUL_MS, iki noktada en az IKI_NOKTA_MS (model virgülü bazen neredeyse geçiyordu).
YEREL_HECE, YEREL_KAT = 4, 1.3
VIRGUL_MS, IKI_NOKTA_MS = 125, 200
# DURAKLAMA SESSİZLİKTİR (03.11.2026, Whisper ile 321 makale cümlesi): noktalamanın süresi uzatılınca model o boşluğu
# sesle dolduruyordu ("İkincisi:" → "ikincisihe", "örnek;" → "örneka", "79 il," → "ile"; baştaki dolguda da aynı, bkz.
# BAS_DOLGU). Noktalama modelde en çok DURAK_KALAN birim kalır, istenen duraklamanın geri kalanı dalgaya sessizlik
# olarak eklenir. Cümle sonunda son harften SON_PAY birim sonra kesilir ("Ne yapıyor?" → "yapıyors" kuyruğu).
# Bedeli: 40 uzun genel cümlede UTMOS 3,07 → 2,99 (ölçü eklenmiş sessizliği sevmiyor; geçiş 5-20 ms, taban gürültüsü,
# noktalamanın doğal süresini korumak, kesimsiz bitiş fark etmedi); okunuş denetiminde takılan aynı (koşudan koşuya ±3).
DURAK_KALAN, SON_PAY, GECIS = 2, 2, 120    # birim, birim, örnek (5 ms sönüm / açılım)
# HARF HARF OKUNAN KISALTMA: harf adları (turkce_normalize.HARF_OKUNUSU) art arda gelince biraz yavaş ve ayrık okunur
HARF_ADLARI = ("a", "be", "ce", "çe", "de", "e", "fe", "ge", "he", "ı", "i", "je", "ka", "le", "me", "ne", "o", "ö",
               "pe", "re", "se", "şe", "te", "u", "ü", "ve", "ye", "ze", "kü", "iks")
HARF_SOZCUK = ("a", "de", "e", "ne", "o", "ye", "be", "ı", "i", "u", "ü", "ö")   # kendisi de sözcük olanlar
HARF_KAT, HARF_ARA_KAT = 1.3, 2.0
HIZ_BANDI = 0.07
HIZ_EN_COK = 1.35
HIZ_EN_AZ_HECE = 6
_SORU_EKI = re.compile(
    r"^m[ɯiuy](s[ɯiuy]n([ɯiuy]z)?|j[ɯiuy][mz]|d[ɯiuy]ɾ?|jd[ɯiuy](m|n|[ɯiuy]z)?|jmɯʃ|jmiʃ|jmuʃ|jmyʃ)?[?.,!]*$")
_UNLU = set("aeɯiouøyœæɛɪʊ")


def model_klasoru(klasor: str | Path | None = None) -> Path:
    adaylar = [Path(klasor)] if klasor else []
    if os.environ.get("SESLEN_TTS_MODEL"):
        adaylar.append(Path(os.environ["SESLEN_TTS_MODEL"]))
    for a in adaylar:
        if all((a / f).exists() for f in DOSYALAR):
            return a
    if os.environ.get("SESLEN_TTS_INDIRME") != "0":
        try:
            from huggingface_hub import snapshot_download
            a = Path(snapshot_download(HF_DEPO, allow_patterns=DOSYALAR))
            if all((a / f).exists() for f in DOSYALAR):
                return a
        except Exception:
            pass
    raise FileNotFoundError(f"ses modeli bulunamadı: Hugging Face'ten ({HF_DEPO}) indirilemedi ve SESLEN_TTS_MODEL "
                            f"ile klasör gösterilmedi. Gereken dosyalar: {', '.join(DOSYALAR)}")


def soru_egrisi(okunus: str, sozluk: dict) -> np.ndarray:
    """Okunuşun modele giren her işaretine yarım ton eki: soru ekinden önceki sözcüğün vurgulu hecesine SORU_TEPE,
    gerisine 0. Uzunluk, sözlükte olan karakter sayısıdır (olmayanlar işarete çevrilmez)."""
    ek = np.zeros(len(okunus), np.float32)
    if "?" in okunus:
        sozcukler, i = [], 0
        for s in okunus.split(" "):
            sozcukler.append((i, s))
            i += len(s) + 1
        for j in range(1, len(sozcukler)):
            bas, s = sozcukler[j]
            if not _SORU_EKI.match(s.replace("ˈ", "")) or "?" not in okunus[bas:]:
                continue
            b0, onceki = sozcukler[j - 1]
            if not onceki or onceki[-1] in "?.,!;:":
                continue
            v = onceki.rfind("ˈ")
            if v < 0:               # vurgu işareti yoksa son hece: son ünlü + önündeki ünsüz
                v = max((k for k, c in enumerate(onceki) if c in _UNLU), default=0)
                v -= v > 0 and onceki[v - 1] not in _UNLU
            # hecenin sonu: vurgulu ünlüden sonraki ilk ayrı ünlünün (bitişik ünlü aynı çekirdek) önündeki ünsüz
            k = next((k for k in range(v, len(onceki)) if onceki[k] in _UNLU), len(onceki))
            while k < len(onceki) and (onceki[k] in _UNLU or onceki[k] == "ː"):
                k += 1
            sonraki = next((m for m in range(k, len(onceki)) if onceki[m] in _UNLU), None)
            son = len(onceki) if sonraki is None else max(sonraki - 1, k)
            ek[b0 + v: b0 + son] = SORU_TEPE
    return np.array([e for c, e in zip(okunus, ek) if c in sozluk], np.float32)


def denetim_klasoru(klasor: Path, indir: bool) -> Path | None:
    """Özdenetimin tanıyıcısı (denetim.onnx, denetim.json; isteğe bağlı, ~360 MB): model klasöründe varsa orası;
    yoksa ve indir ise Hugging Face'ten; bulunamazsa None."""
    from .denetim import DOSYALAR as D
    if all((klasor / f).exists() for f in D):
        return klasor
    if indir and os.environ.get("SESLEN_TTS_INDIRME") != "0":
        try:
            from huggingface_hub import snapshot_download
            a = Path(snapshot_download(HF_DEPO, allow_patterns=D))
            if all((a / f).exists() for f in D):
                return a
        except Exception:
            pass
    return None


def windows_kisitini_kaldir() -> None:
    """Windows arka plandaki süreçleri enerji tasarrufu için yavaşlatıyor (EcoQoS): ölçümde ses üretimi 4-5 kat
    yavaşladı. Bu süreç için kısıt kapatılır; önceliğe dokunulmaz. SESLEN_TTS_SUREC=0 ile kapalı."""
    if os.name != "nt" or os.environ.get("SESLEN_TTS_SUREC") == "0":
        return
    import ctypes
    import ctypes.wintypes as w

    class _Kisit(ctypes.Structure):
        _fields_ = [("Version", w.ULONG), ("ControlMask", w.ULONG), ("StateMask", w.ULONG)]
    k = ctypes.windll.kernel32
    k.OpenProcess.restype = w.HANDLE
    h = k.OpenProcess(0x0200 | 0x0400, False, os.getpid())       # PROCESS_SET_INFORMATION | QUERY_INFORMATION
    if h:
        s = _Kisit(1, 0x1, 0)                                         # PROCESS_POWER_THROTTLING_EXECUTION_SPEED: kapalı
        k.SetProcessInformation(w.HANDLE(h), 4, ctypes.byref(s), ctypes.sizeof(s))
        k.CloseHandle(w.HANDLE(h))


def _yalin(okunus: str) -> str:
    return "".join(c for c in okunus if c.isalpha() and c not in "ˈː")


def harf_kumesi(sure: np.ndarray, okunus: str, semboller: dict, araliklar: list[tuple[int, int]],
                harfler: set[str], sozcuk_harfler: set[str]) -> np.ndarray:
    """Harf harf okunan kısaltma ("te re ce be se"): art arda en az iki harf adı, en az biri sözcük olmayan ("ne de"
    kalır) → harfler HARF_KAT, aralarındaki boşluk HARF_ARA_KAT uzar. Model harf adlarını tek sözcük gibi yapıştırıp
    hızlı söylüyordu ("TR-CBS-Komut" → "terecebese", 03.11.2026)."""
    sozcukler = [_yalin(p) for p in okunus.split(" ") if any(c.isalpha() for c in p if c in semboller)]
    if len(sozcukler) != len(araliklar):
        return sure
    harf = [s in harfler for s in sozcukler]
    sure, i = sure.copy(), 0
    while i < len(harf):
        j = i
        while j < len(harf) and harf[j]:
            j += 1
        if j - i >= 2 and any(sozcukler[k] not in sozcuk_harfler for k in range(i, j)):
            for k in range(i, j):
                a, b = araliklar[k]
                sure[a:b] *= HARF_KAT
                if k < j - 1:                                  # sözcükle sonraki harf arasındaki işaretler
                    sure[b:araliklar[k + 1][0]] *= HARF_ARA_KAT
        i = max(j, i + 1)
    return sure


def duraklari_ayir(sure: np.ndarray, kod: list[int], durak: dict[int, tuple[int, int]]):
    """durak: noktalama kodu → (en az ms, ek ms); istenen duraklama max(doğal, en az) + ek. Noktalamanın modeldeki
    süresi en çok DURAK_KALAN birim kalır; geri kalanı [(işaret sırası, birim)] olarak döner (bkz. sessizlik_ekle)."""
    sure, duraklar = sure.copy(), []
    for i, k in enumerate(kod, 1):
        if k in durak:
            en_az, ek = durak[k]
            istenen = max(int(sure[i]), round(en_az / BIRIM_MS)) + round(ek / BIRIM_MS)
            sure[i] = min(int(sure[i]), DURAK_KALAN)
            if istenen > sure[i]:
                duraklar.append((i, istenen - int(sure[i])))
    return sure, duraklar


def sessizlik_ekle(w: np.ndarray, sure: np.ndarray, duraklar: list[tuple[int, int]], son: int = 0,
                   pay: int = SON_PAY) -> np.ndarray:
    """Dalga: son > 0 ise son harften (işaret sırası son) pay birim sonra kesilir, son birimde söner; duraklar
    noktalamanın ilk biriminden sonra sessizlik olarak eklenir (önceki sözcüğün bırakışı önde, sonraki sözcüğe
    hazırlık arkada kalır; noktalamanın sonuna eklenince tanıyıcı "rrögar", "hhedef" duyuyordu). Kenarlarda GECIS
    örnek sönüm / açılım."""
    sinir = np.cumsum(sure) * ORNEK_BIRIM                  # her işaretin bittiği örnek
    w = np.array(w, dtype=np.float32)
    if son:
        w = w[:min(len(w), int(sinir[son]) + pay * ORNEK_BIRIM)]
        n = min(len(w), ORNEK_BIRIM)
        w[len(w) - n:] *= np.linspace(1, 0, n, dtype=w.dtype)
    parca, bas = [], 0
    for i, birim in duraklar:
        p = int(sinir[i - 1]) + min(int(sure[i]), 1) * ORNEK_BIRIM
        if p >= len(w):
            break
        n = min(GECIS, p - bas, len(w) - p)
        w[p - n:p] *= np.linspace(1, 0, n, dtype=w.dtype)
        w[p:p + n] *= np.linspace(0, 1, n, dtype=w.dtype)
        parca += [w[bas:p], np.zeros(birim * ORNEK_BIRIM, w.dtype)]
        bas = p
    return np.concatenate(parca + [w[bas:]]) if parca else w


def hizala(sure: np.ndarray) -> np.ndarray:
    """İşaret başına süre (birim) → hizalama matrisi (1, L, F): her işaret kendi süresi kadar çerçeveye yayılır."""
    return np.repeat(np.eye(len(sure), dtype=np.float32), sure, axis=1)[None]


class SesModeli:
    def __init__(self, klasor: str | Path | None = None, is_parcacigi: int = 0):
        import onnxruntime as ort
        windows_kisitini_kaldir()
        k = self.klasor = model_klasoru(klasor)
        self.is_parcacigi = is_parcacigi
        se = ort.SessionOptions()
        se.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        se.intra_op_num_threads = is_parcacigi or min(6, os.cpu_count() or 1)
        self.metin = ort.InferenceSession(str(k / "metin.onnx"), se, providers=["CPUExecutionProvider"])
        self.ses = ort.InferenceSession(str(k / "ses.onnx"), se, providers=["CPUExecutionProvider"])
        v = np.load(k / "stiller.npz")
        self.stiller = {a: v[a].astype(np.float32) for a in v.files}
        self.sesler: dict = json.loads((k / "sesler.json").read_text(encoding="utf-8"))
        self.semboller: dict = json.loads((k / "semboller.json").read_text(encoding="utf-8"))
        self._virgul = {self.semboller[c] for c in "," if c in self.semboller}
        self._noktali = {self.semboller[c] for c in ";" if c in self.semboller}
        self._iki_nokta = {self.semboller[c] for c in ":" if c in self.semboller}

    def okunus(self, metin: str) -> str:
        from turkce_vurgu import g2p
        return g2p(metin, vurgu=True)

    def stil(self, ses: str, ton: str = "notr") -> np.ndarray:
        try:
            return self.stiller[f"{ses}/{ton}"]
        except KeyError:
            raise KeyError(f"bilinmeyen ses/ton: {ses}/{ton}; sesler: {', '.join(self.sesler)}") from None

    def _metin_tarafi(self, okunus: str, stil: np.ndarray):
        kod = [self.semboller[c] for c in okunus if c in self.semboller]
        d, t_en, sure = self.metin.run(None, {"tokens": np.array([[0, *kod, 0]], np.int64), "stil": stil})
        return kod, d, t_en, sure

    def hece_hizi(self, metin: str, stil: np.ndarray, hiz: float = 1.0) -> float | None:
        """Modelin bu cümle için kendiliğinden seçeceği hız (hece/sn; noktalama duraklamaları dahil). Kısa cümlede
        (HIZ_EN_AZ_HECE heceden az) None. Seslen.seslendir belgenin hız ortancasını bununla bulur."""
        okunus = self.okunus(metin)
        hece = sum(c in _UNLU for c in okunus)
        if hece < HIZ_EN_AZ_HECE:
            return None
        sure = self._metin_tarafi(okunus, np.asarray(stil, np.float32).reshape(1, -1))[3]
        return hece / max(1e-6, float(sure[1:-1].sum()) / hiz * BIRIM_MS / 1000)

    def sozcuk_araliklari(self, okunus: str) -> list[tuple[int, int]]:
        """Okunuştaki her sözcüğün işaret aralığı [baş, son) (baştaki dolgu 0; noktalama dahil değil)."""
        araliklar, i = [], 1
        for parca in okunus.split(" "):
            bas = son = None
            for c in parca:
                if c not in self.semboller:
                    continue
                if c.isalpha():
                    bas = i if bas is None else bas
                    son = i + 1
                i += 1
            i += " " in self.semboller
            if bas is not None:
                araliklar.append((bas, son))
        return araliklar

    def konus(self, metin: str, stil: np.ndarray, hiz: float = 1.0, perde: float = 1.0, tonlama: float = 1.0,
              virgul_ms: int = 0, hedef_hece: float | None = None, okunus: str | None = None,
              uzat: list[tuple[int, int, float]] | None = None) -> np.ndarray:
        """Tek cümle → 24 kHz dalga (float32). hiz > 1 hızlı; perde: F0 çarpanı; tonlama: perde dalgalanması
        çarpanı (1 doğal); virgul_ms: virgül ve noktalı virgülde ek duraklama. hedef_hece: hız dengesi (hece/sn,
        hiz çarpanı uygulanmış); verilirse cümlenin hızı bunun ±HIZ_BANDI içine çekilir (bkz. HIZ_BANDI).
        okunus / uzat: özdenetimin onarımları (hazır okunuş; [(baş, son, kat)] işaret aralığının süresi çarpılır)."""
        bitis = 0 if okunus is not None else self._kisa_bitis(metin)
        okunus = okunus if okunus is not None else self.okunus(metin + TASIYICI if bitis else metin)
        stil = np.asarray(stil, np.float32).reshape(1, -1)
        kod, d, t_en, sure = self._metin_tarafi(okunus, stil)
        egri = np.concatenate([[0], soru_egrisi(okunus, self.semboller), [0]]).astype(np.float32)[None]
        hece = sum(c in _UNLU for c in okunus)
        if hedef_hece and not bitis and hece >= HIZ_EN_AZ_HECE:
            oran = hece / max(1e-6, float(sure[1:-1].sum()) / hiz * BIRIM_MS / 1000)
            alt, ust = hedef_hece * (1 - HIZ_BANDI), hedef_hece * (1 + HIZ_BANDI)
            if not alt <= oran <= ust:
                hiz *= float(np.clip((ust if oran > ust else alt) / oran, 1 / HIZ_EN_COK, HIZ_EN_COK))
        sure = sure / hiz
        if hedef_hece and not bitis:
            sure = self._yerel_hiz(sure, okunus)
        if not bitis:
            sure = harf_kumesi(sure, okunus, self.semboller, self.sozcuk_araliklari(okunus), *self._harf_adlari())
        for bas, son, kat in uzat or ():
            sure[bas:son] = sure[bas:son] * kat
        sure = np.maximum(np.round(sure), 1).astype(np.int64)
        sure[0] = min(sure[0], BAS_DOLGU)
        # noktalı virgülde doğalın üstüne en az NOKTALI_MS, virgülde en az VIRGUL_MS, iki noktada en az IKI_NOKTA_MS
        durak = {k: (0, max(virgul_ms, NOKTALI_MS)) for k in self._noktali}
        durak |= {k: (VIRGUL_MS, virgul_ms) for k in self._virgul} | {k: (IKI_NOKTA_MS, 0) for k in self._iki_nokta}
        sure, duraklar = duraklari_ayir(sure, kod, durak)
        g = {"d": d, "t_en": t_en, "hiza": hizala(sure), "s": stil[:, 128:], "ref": stil[:, :128],
             "perde": np.array([perde], np.float32), "tonlama": np.array([tonlama], np.float32), "egri": egri}
        w = self.ses.run(None, g)[0][..., :-50].reshape(-1)       # sondaki tık
        # kısa cümlede taşıyıcıdan önce son harften 25 ms sonra, başka cümlede SON_PAY birim sonra kesilir
        return sessizlik_ekle(w, sure, duraklar, bitis or self._son_harf(kod), 1 if bitis else SON_PAY)

    def _son_harf(self, kod: list[int]) -> int:
        """Son harfin işaret sırası (baştaki dolgu 0); harf yoksa 0."""
        if not hasattr(self, "_harfsiz"):
            self._harfsiz = {i for c, i in self.semboller.items() if not c.isalpha()}
        son = len(kod)
        while son and kod[son - 1] in self._harfsiz:
            son -= 1
        return son

    def _harf_adlari(self) -> tuple[set[str], set[str]]:
        """(harf adlarının yalın okunuşları, bunlardan kendisi de sözcük olanlar)."""
        if not hasattr(self, "_harf"):
            self._harf = ({_yalin(self.okunus(a)) for a in HARF_ADLARI}, {_yalin(self.okunus(a)) for a in HARF_SOZCUK})
        return self._harf

    def _yerel_hiz(self, sure: np.ndarray, okunus: str) -> np.ndarray:
        """Cümle içinde birden hızlanan bölüm (en az YEREL_HECE heceli ardışık sözcükler) cümlenin ortalama hızının
        YEREL_KAT katını aşıyorsa o bölüm yavaşlatılır. Ortalama yalnız sözcüklerden (duraklamalar sayılmaz)."""
        isaret = [c for c in okunus if c in self.semboller]
        unlu = np.array([0, *[c in _UNLU for c in isaret], 0], np.float64)
        araliklar = self.sozcuk_araliklari(okunus)
        if not araliklar:
            return sure
        h = sum(unlu[a:b].sum() for a, b in araliklar)
        d = sum(sure[a:b].sum() for a, b in araliklar)
        if h < HIZ_EN_AZ_HECE or d <= 0:
            return sure
        sinir = YEREL_KAT * h / d
        sure, bolum = sure.copy(), []
        for a, b in araliklar + [(None, None)]:
            if a is not None:
                bolum.append((a, b))
            if bolum and (a is None or sum(unlu[x:y].sum() for x, y in bolum) >= YEREL_HECE):
                hb, db = sum(unlu[x:y].sum() for x, y in bolum), sum(sure[x:y].sum() for x, y in bolum)
                if hb >= YEREL_HECE and db > 0 and hb / db > sinir:
                    for x, y in bolum:
                        sure[x:y] *= (hb / db) / sinir
                bolum = []
        return sure

    def _kisa_bitis(self, metin: str) -> int:
        """Kısa cümlede (soru değil) son harfin işaret sırası (baştaki dolgu 0), değilse 0. Bkz. KISA_SOZCUK."""
        metin = metin.strip()
        if len(metin.split()) > KISA_SOZCUK or not metin.endswith((".", "!")):
            return 0
        kod = [self.semboller[c] for c in self.okunus(metin) if c in self.semboller]
        uzun = [self.semboller[c] for c in self.okunus(metin + TASIYICI) if c in self.semboller]
        if uzun[:len(kod)] != kod:
            return 0                                  # okunuş taşıyıcıyla değişiyorsa kesim yeri bilinmez
        nokta = {self.semboller[c] for c in ".!…," if c in self.semboller}
        son = len(kod)
        while son > 1 and kod[son - 1] in nokta:
            son -= 1
        return son
