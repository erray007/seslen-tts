# -*- coding: utf-8 -*-
"""
seslen-tts — işlemcide çalışan, duygulu, vurguyu bilen açık Türkçe ses modeli.

    from seslen_tts import Seslen, wav_yaz
    s = Seslen()                                   # model ilk kullanımda Hugging Face'ten iner
    wav_yaz("merhaba.wav", s.tum("Merhaba! Bugün size ne okumamı istersiniz?", ses="zeynep"))
"""
from .model import ORAN, SesModeli
from .seslen import TONLAR, Parca, Seslen, wav_yaz

__all__ = ["ORAN", "SesModeli", "Seslen", "Parca", "TONLAR", "wav_yaz"]
__version__ = "0.3.0"
