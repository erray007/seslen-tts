# -*- coding: utf-8 -*-
"""
HIZ ÖLÇÜMÜ ORTAK — model klasörü, çalışma klasörü, kaba okunuş, çözücü girdisi, oturum kurma.

Model klasörü: SESLEN_TTS_MODEL (yoksa Hugging Face önbelleği, model.model_klasoru). Ara dosyalar (girdi .npz,
yamalı / nicelenmiş .onnx) araclar/hiz/cikti/ altına yazılır (git dışında).

KABA OKUNUŞ: turkce-vurgu kurulamayan makinede ölçüm için harf harf IPA'ya yakın çeviri. Zamanlama ve iki çözücünün
aynı girdide karşılaştırılması için yeterli; dinleme için değil (vurgu ve normalleştirme yok).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from seslen_tts import model as M   # noqa: E402

CIKTI = Path(__file__).resolve().parent / "cikti"
CIKTI.mkdir(exist_ok=True)
_TR = str.maketrans({"ı": "ɯ", "ş": "ʃ", "ç": "tʃ", "c": "dʒ", "ğ": "ː", "ö": "ø", "ü": "y", "r": "ɾ", "j": "ʒ",
                     "y": "j"})


def kaba(metin: str) -> str:
    return metin.lower().translate(_TR)


def model_klasoru() -> Path:
    return M.model_klasoru()


def ses_modeli(is_parcacigi: int = 0, kaba_okunus: bool = True) -> M.SesModeli:
    """SesModeli; kaba_okunus ise okunus() turkce-vurgu yerine kaba() kullanır."""
    sm = M.SesModeli(None, is_parcacigi)
    if kaba_okunus:
        sm.okunus = kaba
    return sm


def cozucu_girdisi(sm: M.SesModeli, metin: str, ses: str = "nehir", ton: str = "notr") -> dict:
    """ses.onnx'in girdisi (hız / perde / tonlama 1, soru ezgisi 0): yalnız çözücüyü ölçmek için."""
    stil = sm.stil(ses, ton).reshape(1, -1)
    _, d, t_en, sure = sm._metin_tarafi(kaba(metin), stil)
    sure = np.maximum(np.round(sure), 1).astype(np.int64)
    return {"d": d, "t_en": t_en, "hiza": M.hizala(sure), "s": stil[:, 128:], "ref": stil[:, :128],
            "perde": np.array([1.], np.float32), "tonlama": np.array([1.], np.float32),
            "egri": np.zeros((1, len(sure)), np.float32)}


def oturum(yol, is_parcacigi: int = 4, **ayar):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = is_parcacigi
    for k, v in ayar.items():
        setattr(so, k, v)
    return ort.InferenceSession(str(yol), so, providers=["CPUExecutionProvider"])
