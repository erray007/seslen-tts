# -*- coding: utf-8 -*-
"""
HUGGING FACE'E YÜKLE — model klasörünü (Seslen'de ogrenci/acikAktar.py üretir) ve bu klasördeki model kartını
seslen-tr/seslen-tts deposuna koyar.

Önce: pip install huggingface_hub && huggingface-cli login   (yazma yetkili kendi anahtarınız; depoda/sohbette değil)
    python hf/yukle.py <model klasörü>            gizli depo (önce kendiniz bakın)
    python hf/yukle.py <model klasörü> --acik     herkese açık
"""
import argparse
import os
import shutil
import tempfile
from pathlib import Path

DEPO = os.environ.get("SESLEN_TTS_HF", "seslen-tr/seslen-tts")
DOSYALAR = ["metin.onnx", "ses.onnx", "stiller.npz", "sesler.json", "semboller.json"]
ISTEGE_BAGLI = ["denetim.onnx", "denetim.json"]   # özdenetim tanıyıcısı (seslen_tts.denetim), ~360 MB
URUN_SESLERI = {"damla", "kuzey"}           # Seslen'in yerel ürün sesi adları: açık modelde geçmez


def main() -> None:
    import numpy as np
    from huggingface_hub import HfApi
    ap = argparse.ArgumentParser()
    ap.add_argument("klasor", type=Path)
    ap.add_argument("--acik", action="store_true", help="depoyu herkese açık yap (varsayılan gizli)")
    a = ap.parse_args()
    eksik = [f for f in DOSYALAR if not (a.klasor / f).exists()]
    if eksik:
        raise SystemExit("eksik dosya: " + ", ".join(eksik))
    sesler = {k.split("/")[0] for k in np.load(a.klasor / "stiller.npz").files}
    if sesler & URUN_SESLERI:
        raise SystemExit(f"ürün sesi var, yüklenmedi: {sesler & URUN_SESLERI}")
    with tempfile.TemporaryDirectory() as t:
        for f in DOSYALAR + [f for f in ISTEGE_BAGLI if (a.klasor / f).exists()]:
            shutil.copy(a.klasor / f, Path(t) / f)
        if (a.klasor / "ornekler").exists():
            shutil.copytree(a.klasor / "ornekler", Path(t) / "ornekler")
        shutil.copy(Path(__file__).with_name("README.md"), Path(t) / "README.md")
        api = HfApi()
        api.create_repo(DEPO, private=not a.acik, exist_ok=True)
        api.upload_folder(folder_path=t, repo_id=DEPO, commit_message=os.environ.get("SESLEN_TTS_MESAJ", "model, örnekler ve model kartı"))
    print(f"yüklendi: https://huggingface.co/{DEPO} ({'açık' if a.acik else 'gizli'}; sesler: {', '.join(sorted(sesler))})")


if __name__ == "__main__":
    main()
