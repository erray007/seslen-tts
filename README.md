# seslen-tts

> **English:** An open Turkish text-to-speech model that runs on CPU, without PyTorch and without internet: 6 voices × 9
> emotions, correct Turkish lexical stress and question intonation, automatic emotion per sentence and reading style per
> text. ONNX Runtime only (~330 MB model, downloaded from Hugging Face on first use). Real-time factor ≈ 0.2–0.4 on a
> laptop CPU.

**İşlemcide çalışan, duygulu, vurguyu bilen açık Türkçe ses modeli.** GPU, internet ve PyTorch gerekmez.

```python
from seslen_tts import Seslen, wav_yaz

s = Seslen()                                    # model ilk kullanımda Hugging Face'ten iner (~330 MB)
wav_yaz("merhaba.wav", s.tum("Merhaba! Yarın bize gelecek misiniz? Dikkat edin, yol kaygan.", ses="zeynep"))
```

```
seslen-tts "Bir varmış bir yokmuş, ormanda meraklı bir sincap yaşarmış." -s ayse -o masal.wav
seslen-tts --sesler
```

## Neler yapar

- **6 ses, 9 duygu:** nötr, neşeli, heyecan, uyarı, merak, özür, üzgün, kızgın, fısıltı.
- **Türkçe vurgu doğru okunur:** "ANkara", "geLİyor", "GELmedi". Okunuş [turkce-vurgu](https://github.com/seslen-tr/turkce-vurgu)
  ile çıkarılır; sayılar, tarihler, saatler, para ve kısaltmalar [turkce-normalize](https://github.com/seslen-tr/turkce-normalize)
  ile okunur hâle gelir ("14:30'da" → "on dört otuzda").
- **Soru ezgisi:** evet/hayır sorusunda ezgi soru ekinden önceki sözcüğün vurgulu hecesinde yükselir
  ("yardımcı olaBİLİR misin?").
- **Otomatik duygu:** her cümlenin tonu metinden seçilir: soru merakla, uyarı uyarıyla, sevinç neşeyle okunur.
  [turkce-duygu](https://github.com/seslen-tr/turkce-duygu) gerekir (`pip install "seslen-tts[duygu]"`).
- **Otomatik üslup:** haber tempolu ve düz, masal yavaş ve hareketli, tarif adım adım okunur.
  [turkce-uslup](https://github.com/seslen-tr/turkce-uslup) gerekir (`pip install "seslen-tts[uslup]"`).

## Kurulum

```
pip install seslen-tts               # temel: sesler, vurgu, soru ezgisi
pip install "seslen-tts[duygu]"      # + otomatik duygu (turkce-duygu)
pip install "seslen-tts[uslup]"      # + otomatik üslup (turkce-uslup)
pip install "seslen-tts[hepsi]"      # ikisi birden
pip install git+https://github.com/seslen-tr/seslen-tts   # en son kaynak, GitHub'dan
```

Python 3.9+, Windows / Linux / macOS. Bağımlılıklar: `numpy`, `onnxruntime`, `huggingface_hub`, `turkce-vurgu[tam]`
(o da metni [turkce-normalize](https://github.com/seslen-tr/turkce-normalize) ile hazırlar).

### Model dosyaları

Model beş dosyadır: `metin.onnx`, `ses.onnx`, `stiller.npz`, `sesler.json`, `semboller.json`. Şu sırayla aranır:

1. `Seslen(klasor="...")` ile verilen klasör,
2. `SESLEN_TTS_MODEL` ortam değişkenindeki klasör,
3. Hugging Face deposu (varsayılan `seslen-tr/seslen-tts`, `SESLEN_TTS_HF` ile değişir): ilk kullanımda iner,
   önbellekte kalır. `SESLEN_TTS_INDIRME=0` indirmeyi kapatır.

Windows'ta enerji kısıtını kaldırma `SESLEN_TTS_SUREC=0` ile kapatılır. İş parçacığı sayısı
`Seslen(is_parcacigi=4)` ile verilir (varsayılan: çekirdek sayısı, en çok 6).

## Sesler

| Ses | Kim | Doğallık (UTMOS) |
|---|---|---|
| `ayse` | olgun kadın (~45), orta-kalın, güven veren | 3,89 |
| `koca` | iri kalın dev, gürleyen ama sevimli (çizgi karakter) | 3,89 |
| `hasan` | yaşlı erkek (~72), kalın, bilge dede | 3,26 |
| `zeynep` | genç kız (~17), ince, canlı | 3,25 |
| `emre` | genç erkek (~18), orta, enerjik | 2,87 |
| `fatma` | yaşlı kadın (~70), ince, şefkatli nine | 2,38 |

Örnek sesler: [Hugging Face model sayfası](https://huggingface.co/seslen-tr/seslen-tts/tree/main/ornekler).

## Kullanım

```python
s = Seslen()
for p in s.seslendir(metin, ses="emre", ton="otomatik", uslup="otomatik"):   # cümle cümle, akış hâlinde
    print(p.ton, p.cumle)            # p.ses: 24 kHz float32 dizi; p.sessizlik_ms: sonraki duraklama

s.tum(metin, ses="fatma", ton="uzgun")        # bütün metin tek tonda
s.tum(metin, ses="hasan", hiz=0.9, perde=1.05) # hız ve ses perdesi
s.tum(metin, ses="zeynep", ton="neseli", yogunluk=0.6)   # duygu şiddeti (0-1, 1 = tam ton)
```

- `ton`: `"otomatik"` ya da `notr`, `neseli`, `heyecan`, `uyari`, `merak`, `ozur`, `uzgun`, `kizgin`, `fisilti`
  (`seslen_tts.TONLAR`). Bir sesin tonları: `s.tonlar("zeynep")`; sesler ve açıklamaları: `s.sesler`.
- `uslup`: `"otomatik"` ya da `haber`, `bilgi`, `talimat`, `deneme`, `anlati`, `masal`, `sohbet`. Üslup hızı,
  ezgi genişliğini ve duraklamaları belirler; `hiz` verilirse üslubun hızını ezer.
- turkce-duygu kurulu değilse her cümle nötr, turkce-uslup kurulu değilse üslup düz okunur.
- `wav_yaz(yol, dalga)` 16 bit tek kanal 24 kHz WAV yazar (standart `wave` modülüyle; ses tepesi 0,9'a ölçeklenir).

Alt düzey: `s.model.konus(cumle, s.model.stil("ayse", "merak"))` tek cümleyi seslendirir.

### Komut satırı

```
seslen-tts "metin" [-s SES] [-t TON] [-u USLUP] [--hiz 1.1] [--perde 1.0] [-o cikti.wav]
seslen-tts --sesler                  # sesleri ve tonlarını listele
echo "Merhaba!" | seslen-tts -o merhaba.wav     # metin verilmezse standart girdiden
```

Varsayılanlar: ses `zeynep`, ton ve üslup `otomatik`, çıktı `seslen.wav`. `python -m seslen_tts` aynı işi yapar.

## Ölçümler

- **Okunuş denetimi** (Türkçe wav2vec2 CTC + sözcük başına GOP, 6 ses × 14 cümle; sayı, saat ve para dahil): 84
  cümlede 1 sözcük hatası (Hasan: "aradığın" → "aradığım").
- **Doğallık:** UTMOS 2,4–3,9 (tablo; 6 ses × 14 cümle, paketin kendi yoluyla). UTMOS İngilizce dinleyicilerle eğitildi; Türkçe vurguyu yargılayamaz,
  yapaylığı yakalar.
- **Hız** (i7-12700H, 6 iş parçacığı): gerçek zaman oranı ≈ 0,2–0,4. Yani 5 saniyelik cümle 1–2 saniyede üretilir.
  Windows'un arka plan süreçlerine uyguladığı enerji kısıtı paket tarafından kapatılır (yoksa 4–5 kat yavaşlıyordu).

## Nasıl yapıldı

[StyleTTS2](https://github.com/yl4579/StyleTTS2) (iSTFTNet çözücü) öğrenci modeli. Ses ve duygu verisini öğretmen model
[VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) üretti: sesler tariften tasarlandı, gerçek bir kişinin sesi değildir.
Eğitim üç aşamalıdır: temel, difüzyon ve çekişmeli ince ayar, son olarak vurgu işaretli okunuş ile
[Türkçe PL-BERT](https://huggingface.co/seslen-tr/turkce-plbert). Eğitim tek RTX 4090'da yapıldı, son ince ayar
(0.2.0) tek A100'de; bu turda her eğitim kaydı otomatik okunuş denetiminden (Türkçe wav2vec2 CTC) geçti. Ağlar ONNX'e
aktarıldı; aktarımın sesi değiştirmediği sınandı (log-mel farkı, modelin kendi iki çalıştırması arasındaki farkın altında).

## Sınırlılıklar

- **Ses klonlama yok.** Bu bilerek böyle: açık modelle başkasının sesini taklit etmek mümkün olmasın diye yalnız hazır
  sesler var.
- Yalnız Türkçe. Yabancı sözcükler Türkçe okunuşla okunur.
- Uzun yabancı adlarda, kısaltma yığınlarında ve çok uzun cümlelerde okunuş hataları olabilir.
- Duygular seslere göre farklı belirginliktedir (ölçülenler arasında en belirgin: Ayşe ve Zeynep).

## Lisans ve kullanım koşulları

- **Kod:** MIT ([LICENSE](LICENSE)).
- **Model ağırlıkları ve sesler:** CC BY-SA 4.0. Türkçe PL-BERT Vikipedi metniyle eğitildi; öğretmen VoxCPM2
  Apache-2.0, StyleTTS2 MIT. Telaffuz için yardımcı veri olarak YODAS2 Türkçe kesitleri (Creative Commons lisanslı
  YouTube videoları, CC BY 3.0; Li vd. 2023, "YODAS: YouTube-Oriented Dataset for Audio and Speech") kullanıldı;
  seslerin hiçbiri bu konuşmacılardan değildir, hepsi öğretmenin tarif tasarımıdır.
- **Kullanım koşulu (öğretmen modelin ve StyleTTS2'nin koşullarından):**
  - Üretilen sesin yapay zekâ sesi olduğu dinleyiciye belirtilmelidir.
  - Gerçek bir kişiyi taklit etmek, dolandırıcılık ve dezenformasyon için kullanmak yasaktır.

Bütün ayarlarıyla stüdyo (16 ses, özdenetim, öğretmen kaydı): [Seslen](https://github.com/seslen-tr/seslen) (MIT).

## Düzen ve testler

| Dosya | İçerik |
|---|---|
| `seslen_tts/seslen.py` | `Seslen`: cümlelere bölme, otomatik duygu ve üslup, `seslendir` / `tum`, `wav_yaz` |
| `seslen_tts/model.py` | `SesModeli`: ONNX oturumları, model klasörünü bulma/indirme, soru ezgisi, hizalama |
| `seslen_tts/__main__.py` | komut satırı (`seslen-tts`) |
| `hf/README.md`, `hf/yukle.py` | Hugging Face model kartı ve yükleme betiği (varsayılan gizli depo) |
| `tests/test_model.py` | soru ezgisi, hizalama, cümle bölme; model varsa uçtan uca seslendirme |

```
pytest -q                            # modelsiz testler
SESLEN_TTS_MODEL=<model klasörü> pytest -q   # + seslendirme testi
```

## Kardeş projeler

[turkce-normalize](https://github.com/seslen-tr/turkce-normalize) · [turkce-vurgu](https://github.com/seslen-tr/turkce-vurgu) ·
[turkce-duygu](https://github.com/seslen-tr/turkce-duygu) · [turkce-uslup](https://github.com/seslen-tr/turkce-uslup) ·
[turkce-plbert](https://github.com/seslen-tr/turkce-plbert) · [turkce-tts-sinav](https://github.com/seslen-tr/turkce-tts-sinav)
