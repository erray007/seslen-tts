# Değişiklikler

## 0.3.0 — 16 ses

- Açık model 6 sesten 16 sese çıktı: yeni sesler Nehir, Ege, Serdar, Selin (yetişkin ve genç), Elif, Can, Ece
  (çocuk), Pıtırcık, Fırfır, Kara (çizgi karakter). Ağırlıklar 0.2'dekiyle aynı (Seslen t10); yalnız `stiller.npz`
  ve `sesler.json` büyüdü. Eski paket sürümleri de yeni sesleri kullanabilir.
- Ölçüm (paketin kendi yoluyla, 16 ses × 14 cümle): okunuş denetiminde 224 cümlede 6 sözcük hatası.

## 0.2.1 — kısa cümle

- Tek başına 1-2 sözcüklük cümle ("Evet.", "Eyvah!", "dedi.") arkasına taşıyıcı cümle eklenerek üretilir, ses son
  harfte kesilir: model sona uydurma hece ekliyordu ("evetnı", "dikkatsınız"). 6 ses × 15 kısa cümlede takılan 0.
- Yabancı adlar turkce-normalize 0.1.3 ile Türkçe okunuşla okunur ("Michael Jackson" → Maykıl Ceksın).

## 0.2.0 — yeni model

- Model yenilendi (Seslen t10): sesler yeni öğretmen verisiyle ince ayarlandı; her eğitim kaydı otomatik okunuş
  denetiminden (Türkçe wav2vec2 CTC + sözcük başına GOP) geçti. Ses renkleri değişti. Model dosyalarının adı ve
  biçimi aynı; 0.1.0 paketi de yeni modelle çalışır.
- Kısa ünlem ya da cümle sonrakine virgülle bağlanır (`akici`): tek başına "Dikkat!" ya da "Merhaba!" okunurken
  sona uydurma hece ekleniyordu.
- Noktalı virgülde en az 200 ms duraklama (önceden virgülden ayırt edilmiyordu).

## 0.1.0 — ilk açık sürüm

- 6 ses × 9 duygu, PyTorch'suz (ONNX Runtime) çıkarım, soru ezgisi, otomatik duygu ve üslup (isteğe bağlı).
- `Seslen.seslendir` (cümle cümle akış), `Seslen.tum`, `wav_yaz`; hız, perde ve duygu yoğunluğu ayarı.
- Komut satırı: `seslen-tts` / `python -m seslen_tts` (`--sesler`, `-s`, `-t`, `-u`, `--hiz`, `--perde`, `-o`).
- Model Hugging Face'ten (`seslen-tr/seslen-tts`) ilk kullanımda iner; `SESLEN_TTS_MODEL` ile yerel klasör.
- Windows'ta arka plan enerji kısıtı (EcoQoS) süreç için kapatılır.
- Hugging Face model kartı (`hf/README.md`): 6 sesin dinleme örnekleri.
