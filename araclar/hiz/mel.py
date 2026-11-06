# -*- coding: utf-8 -*-
"""
MEL FARKI — çözücü kopyalarının özgün ses.onnx'e log-mel uzaklığı (dB, sesli bölge) ve süresi, 16 sınama girdisinde.
Kıyas çizgisi: özgün modelin kendi iki çalıştırması (kaynak gürültüsü yüzünden; 06.11.2026: 0,51 dB).

    python araclar/hiz/mel.py cikti/sabit.onnx cikti/q_guvenli.onnx …

Dalga SNR'ı burada işe yaramaz: perde azıcık kayınca faz kayar (yalnız Gemm'i nicelemek bile SNR −2 dB veriyordu).
"""
from __future__ import annotations

import sys
import time

import numpy as np

from ortak import CIKTI, model_klasoru, oturum


def _melfb(sr: int = 24000, n: int = 1024, m: int = 80) -> np.ndarray:
    hz = lambda x: 700 * (10 ** (x / 2595) - 1)              # noqa: E731
    mel = lambda f: 2595 * np.log10(1 + f / 700)             # noqa: E731
    b = np.floor((n + 1) * hz(np.linspace(mel(0), mel(sr / 2), m + 2)) / sr).astype(int)
    fb = np.zeros((m, n // 2 + 1))
    for i in range(1, m + 1):
        fb[i - 1, b[i - 1]:b[i]] = (np.arange(b[i - 1], b[i]) - b[i - 1]) / max(1, b[i] - b[i - 1])
        fb[i - 1, b[i]:b[i + 1]] = (b[i + 1] - np.arange(b[i], b[i + 1])) / max(1, b[i + 1] - b[i])
    return fb


_FB = _melfb()


def logmel(w: np.ndarray, n: int = 1024, hop: int = 256) -> np.ndarray:
    w = np.pad(w, (0, max(0, n - len(w))))
    f = np.stack([w[i * hop:i * hop + n] * np.hanning(n) for i in range(1 + (len(w) - n) // hop)])
    return 10 * np.log10(_FB @ (np.abs(np.fft.rfft(f, axis=1)) ** 2).T + 1e-8)


def fark(a: np.ndarray, b: np.ndarray) -> float:
    """Ortak uzunlukta, en yüksekten 60 dB içindeki mel hücrelerinde ortalama mutlak fark."""
    n = min(len(a), len(b))
    A, B = logmel(a[:n]), logmel(b[:n])
    return float(np.abs(A - B)[np.maximum(A, B) > A.max() - 60].mean())


if __name__ == "__main__":
    testler = [dict(np.load(f)) for f in sorted(CIKTI.glob("test_*.npz"))]
    if not testler:
        sys.exit("önce: python araclar/hiz/girdiler.py")
    oz = oturum(model_klasoru() / "ses.onnx")
    r1 = [oz.run(None, g)[0] for g in testler]
    r2 = [oz.run(None, g)[0] for g in testler]
    print(f"{'özgün ↔ özgün (gürültü)':30} mel farkı {np.mean([fark(a, b) for a, b in zip(r1, r2)]):.2f} dB")
    for yol in sys.argv[1:]:
        s = oturum(yol)
        s.run(None, testler[0])
        t0 = time.perf_counter()
        y = [s.run(None, g)[0] for g in testler]
        t = time.perf_counter() - t0
        f = [fark(a, b) for a, b in zip(r1, y)]
        print(f"{yol:30} mel farkı {np.mean(f):.2f} dB (en kötü {max(f):.2f})  süre {t:.1f} sn")
