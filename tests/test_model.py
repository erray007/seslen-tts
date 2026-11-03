import os

import numpy as np
import pytest

from seslen_tts.model import SORU_TEPE, hizala, soru_egrisi
from seslen_tts.seslen import UZUN_CUMLE, _belge_tonu, _bolumler, _cumleler, akici


class _Hepsi(dict):
    def __contains__(self, k):
        return True


def _tepe(okunus):
    e = soru_egrisi(okunus, _Hepsi())
    return "".join(c for c, x in zip(okunus, e) if x > 0)


def test_soru_tepesi_vurgulu_hecede():
    assert _tepe("baˈna jaɾdɯmˈdʒɯ oɫabiˈliɾ misin?") == "ˈliɾ"      # olaBİLİR misin
    assert _tepe("ˈankaɾa ɟyˈzel mi?") == "ˈzel"                     # güZEL mi
    assert _tepe("buˈnu ˈsen mi japˈtɯn?") == "ˈsen"                 # SEN mi yaptın
    assert _tepe("ɟeˈlijoɾ musunuz?") == "ˈli"                       # geLİyor musunuz


def test_soru_sozcuklu_ve_duz_cumleye_dokunmaz():
    assert _tepe("neˈden ˈɟelmedin?") == ""
    assert _tepe("mɯˈsɯɾ ˈtʃok ɟyzelˈdi.") == ""                     # "Mısır" soru eki değil


def test_egri_sozlukte_olmayani_atlar():
    e = soru_egrisi("ab mi?", {"a": 1, "b": 2, " ": 3, "m": 4, "i": 5})   # "?" sözlükte yok
    assert len(e) == 5 and e.max() == SORU_TEPE


def test_hizala():
    h = hizala(np.array([2, 1, 3]))
    assert h.shape == (1, 3, 6)
    assert h[0].sum(axis=0).tolist() == [1] * 6 and h[0].sum(axis=1).tolist() == [2, 1, 3]


def test_cumleler():
    assert _cumleler("Merhaba! Nasılsın? İyiyim.") == ["Merhaba, nasılsın?", "İyiyim."]
    assert _cumleler("Bugün hava güzel. Yarın yağmur var.") == ["Bugün hava güzel.", "Yarın yağmur var."]


MAKALE = """<!-- not -->
# Başlık

Bu yazıda **altı modeli** ölçtüm. Sonuçlar [burada](https://ornek.com/x).

| Model | Doğruluk |
|---|---|
| a | %81,2 |

#EVREN #LLM
"""


def test_belge_bolumleri():
    b = _bolumler(MAKALE)
    metin = " ".join(c for c, _ in b)
    assert not any(x in metin for x in ("<!--", "#", "**", "|", "not"))
    assert b[0] == ("Başlık.", True)
    assert ("Sonuçlar burada.", True) in b and ("Bu yazıda altı modeli ölçtüm.", False) in b
    assert ("a: Doğruluk, %81,2.", True) in b


def test_uzun_cumle_bolunur():
    uzun = "Bu çok uzun bir cümle, " * 20 + "bitti."
    assert all(len(c) <= UZUN_CUMLE for c, _ in _bolumler(uzun))


def test_belge_tonu():
    assert _belge_tonu("kizgin", 0.9) == ("notr", 0.0)            # belgede güçlü duygu yok
    assert _belge_tonu("merak", 0.3) == ("notr", 0.0)             # belirsizse nötr
    assert _belge_tonu("merak", 0.8) == ("merak", 0.8)


def test_akici():
    assert akici("Dikkat! Bu işlem geri alınamaz.") == "Dikkat, bu işlem geri alınamaz."
    assert akici("Evet! Tamam! Gidelim.") == "Evet, tamam, gidelim."
    assert akici("Dr. Ahmet geldi.") == "Dr. Ahmet geldi."               # kısaltma
    assert akici("Madde 3. Sonra bakarız.") == "Madde 3. Sonra bakarız."  # sayı
    assert akici("Tamam! ABD'ye gidiyoruz.") == "Tamam, ABD'ye gidiyoruz."


@pytest.mark.skipif(not os.environ.get("SESLEN_TTS_MODEL"), reason="model yok (SESLEN_TTS_MODEL)")
def test_seslendirme():
    from seslen_tts import ORAN, Seslen
    s = Seslen()
    assert "notr" in s.tonlar("zeynep")
    parcalar = list(s.seslendir("Bugün hava güzel. Yarın gelecek misiniz?", ses="zeynep", ton="notr"))
    assert len(parcalar) == 2
    assert all(0.4 < len(p.ses) / ORAN < 5 and np.isfinite(p.ses).all() for p in parcalar)


@pytest.mark.skipif(not os.environ.get("SESLEN_TTS_MODEL"), reason="model yok (SESLEN_TTS_MODEL)")
def test_kisa_cumle_tasiyicisiz_uzunlukta():
    from seslen_tts import ORAN, Seslen
    s = Seslen()
    assert s.model._kisa_bitis("Evet.") > 0 and s.model._kisa_bitis("Evet?") == 0
    assert s.model._kisa_bitis("Bugün hava çok güzel.") == 0
    w = s.model.konus("Evet.", s.model.stil("zeynep"))
    assert 0.2 < len(w) / ORAN < 1.2                      # taşıyıcı cümle kesildi


@pytest.mark.skipif(not os.environ.get("SESLEN_TTS_MODEL"), reason="model yok (SESLEN_TTS_MODEL)")
def test_hiz_dengesi_ve_belge():
    from seslen_tts import ORAN, Seslen
    from seslen_tts.model import _UNLU
    s = Seslen()
    uzun = ("Soruların bir kısmı kurallarla, bir kısmı açık lisanslı modellerle üretildi; hepsi ikinci bir modelle "
            "ve kurallarla doğrulandı, adayların yüzde seksen dördü kabul edildi.")
    hedef = s.dogal_hiz("zeynep", "notr")
    w = s.model.konus(uzun, s.model.stil("zeynep"), hedef_hece=hedef)
    hece = sum(c in _UNLU for c in s.model.okunus(uzun))
    assert hece / (len(w) / ORAN) < hedef * 1.15          # uzun cümle doğal hızın bandında (eskiden 1,4 katı)
    parcalar = list(s.seslendir(MAKALE, ses="zeynep"))
    assert parcalar and all(p.ton not in ("kizgin", "heyecan", "uzgun", "fisilti") for p in parcalar)


def test_yeniden_ornekle():
    from seslen_tts.denetim import yeniden_ornekle
    t = np.arange(24000) / 24000
    x = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    y = yeniden_ornekle(x, 24000)
    assert len(y) == 16000
    beklenen = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)
    assert np.abs(y[200:-200] - beklenen[200:-200]).max() < 1e-2


def test_uzun_cumle_virgulden_parcalara():
    from seslen_tts.seslen import PARCA_SOZCUK, _virgulden_bol
    c = ("Soruların bir kısmı kurallarla, bir kısmı açık lisanslı modellerle üretildi; hepsi ikinci bir modelle "
         "ve kurallarla doğrulandı, adayların yüzde seksen dördü kabul edildi.")
    p = _virgulden_bol(c)
    assert len(p) > 1 and " ".join(p) == c
    assert all(len(x.split()) <= PARCA_SOZCUK for x in p)


@pytest.mark.skipif(not os.environ.get("SESLEN_TTS_MODEL"), reason="model yok (SESLEN_TTS_MODEL)")
def test_ozdenetim():
    from pathlib import Path
    if not (Path(os.environ["SESLEN_TTS_MODEL"]) / "denetim.onnx").exists():
        pytest.skip("tanıyıcı yok (denetim.onnx)")
    from seslen_tts import Seslen
    s = Seslen()
    assert s.denetim
    p = list(s.seslendir("Bugün hava çok güzel, parka gidelim mi?", ses="zeynep", ton="notr"))
    assert p and "ceza" in p[0].bilgi["denetim"]
