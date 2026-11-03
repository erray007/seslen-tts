# Değişiklikler

## 0.3.0 — 16 ses, makale okuma

Makale okuma (03.11.2026: gerçek makaleler seslendirildi; Markdown işaretleri okunuyor, duygu cümle cümle sıçrıyor,
hız 4,6–7,3 hece/sn oynuyor, kısaltma ve model adları okunmuyor, tablo modeli çökertiyordu):
- **Belge okuyucu:** Markdown / HTML metin cümle yapısına çevrilir (turkce-normalize 0.1.4 `paragraflar`): yorum, kod,
  etiket satırı atılır; başlık, liste maddesi, tablo satırı kendi cümlesi; numaralı madde "Birincisi: …"; paragraf
  sonunda iki kat duraklama. 220 karakterden uzun cümle bölünür (önceden çöküyordu).
- **Belgede duygu sabit:** haber, bilgi, deneme, talimat üslubunda heyecan / kızgın / üzgün / fısıltı seçilmez,
  belirsiz duygu nötr, kısa cümle (başlık) nötr, ton en çok %60. Otomatik yoğunluk 1'i aşmaz (önceden 1,5'e çıkıp
  tonu abartıyordu).
- **Hız dengesi:** model uzun cümleyi hızlı okuyordu; her cümle sesin doğal hızının (`Seslen.dogal_hiz`) ±%7
  bandına çekilir (`SesModeli.konus(hedef_hece=…)`). Bir makalede 4,6–7,7 → 5,4–6,5 hece/sn.
- **Özdenetim** (`Seslen(denetim=…)`, komut satırında `--denetim`): her cümle Türkçe CTC tanıyıcıyla dinlenir,
  bozuk okunan sözcük onarılır (sözcüğü yavaşlatma, ünlüyü uzatma, vurguyu kaldırma, öncesine duraklama, nötr ton).
  PyTorch gerekmez: tanıyıcı (wav2vec2 XLS-R 300M, Common Voice 8 Türkçe, Apache-2.0) ONNX'e aktarılıp 8 bite
  indirildi (`denetim.onnx`, ~360 MB, isteğe bağlı iner; model klasöründe varsa kendiliğinden açık).
- **Uzun cümle** (12 sözcükten uzun) virgül / noktalı virgül / iki noktadan parçalara bölünüp ayrı okunur: model
  uzun cümlede sesleri kaydırıyordu.
- **Duraklama:** virgülde en az 125 ms, iki noktada 200 ms, cümle sonunda 450 ms (model virgülü bazen geçiyordu).
  Duraklama dalgaya **sessizlik** olarak eklenir; noktalamanın süresi modelde 50 ms'yi aşmaz. Süre uzatılınca model o
  boşluğu sesle dolduruyordu: Whisper'la dinlenen 321 makale cümlesinde "İkincisi:" → "ikincisihe", "örnek;" →
  "örneka", "79 il," → "ile". Bedeli: 40 uzun genel cümlede UTMOS 3,07 → 2,99 (ölçü eklenmiş sessizliği sevmiyor);
  okunuş denetimi aynı.
- **Cümle sonu:** dalga son harften 50 ms sonra kesilip söner (sondaki uydurma ses: "Ne yapıyor?" → "yapıyors").
  Kısa cümlenin taşıyıcısı "Hem bunu…" ile başlar; "Bunu…" ile dudak sesi sızıyordu ("Katkı." → "katkım"; 4 ses ×
  20 kısa cümlede doğru duyulan 49 → 71).
- **Özdenetimde stil sapması:** süre ve vurgu onarımlarının düzeltemediği ünlü kaymasında ("giriş" → "geriş") stile
  küçük, tohumu sabit sapma verilip yeniden okunur (iki deneme).
- **Yerel hız:** cümlenin bir bölümü cümle ortalamasının 1,3 katından hızlıysa yavaşlatılır.
- `-d / --dosya`: Markdown, HTML ya da düz metin dosyası okunur.
- turkce-normalize 0.1.4 ve turkce-vurgu 0.1.2: teknik kısaltma ("LLM" → elelem), "EVREN" sözcük gibi, sözcük içi tire, sürüm, model adı,
  birim, İngilizce teknik terimler.

16 ses:

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
