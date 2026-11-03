# -*- coding: utf-8 -*-
"""
    python -m seslen_tts "Merhaba! Yarın bize gelecek misiniz?" -s zeynep -o merhaba.wav
    python -m seslen_tts --sesler
    python -m seslen_tts -d makale.md -s nehir --denetim -o makale.wav
"""
import argparse
import sys
import time

from . import ORAN, Seslen, wav_yaz


def main() -> int:
    ap = argparse.ArgumentParser(prog="seslen_tts", description="Türkçe metni sese çevirir (işlemcide, internetsiz).")
    ap.add_argument("metin", nargs="?", help="okunacak metin (yoksa standart girdiden)")
    ap.add_argument("-s", "--ses", default="zeynep")
    ap.add_argument("-t", "--ton", default="otomatik", help="otomatik ya da notr, neseli, heyecan, uyari, merak, ozur, "
                                                         "uzgun, kizgin, fisilti")
    ap.add_argument("-u", "--uslup", default="otomatik", help="otomatik ya da haber, bilgi, talimat, deneme, anlati, "
                                                           "masal, sohbet")
    ap.add_argument("--hiz", type=float, default=None)
    ap.add_argument("--perde", type=float, default=1.0)
    ap.add_argument("-o", "--cikti", default="seslen.wav")
    ap.add_argument("--sesler", action="store_true", help="sesleri listele")
    ap.add_argument("-d", "--dosya", help="okunacak dosya (Markdown, HTML ya da düz metin)")
    ap.add_argument("--denetim", action="store_true",
                    help="özdenetim: her cümle dinlenir, bozuk sözcük onarılır (tanıyıcı yoksa ~360 MB iner)")
    ap.add_argument("--denetimsiz", action="store_true", help="özdenetimi kapat")
    a = ap.parse_args()
    s = Seslen(denetim=False if a.denetimsiz else (True if a.denetim else None))
    if a.sesler:
        for ad, b in s.sesler.items():
            print(f"{ad:8s} {b['gorunen']:8s} {b['aciklama']}  [{', '.join(b['tonlar'])}]")
        return 0
    if a.dosya:
        from pathlib import Path
        metin = Path(a.dosya).read_text(encoding="utf-8")
    else:
        metin = a.metin or sys.stdin.read()
    t0 = time.perf_counter()
    dalga = s.tum(metin, ses=a.ses, ton=a.ton, uslup=a.uslup, hiz=a.hiz, perde=a.perde)
    wav_yaz(a.cikti, dalga)
    sure = len(dalga) / ORAN
    print(f"{a.cikti}: {sure:.1f} sn ses, {time.perf_counter() - t0:.1f} sn'de", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
