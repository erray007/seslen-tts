---
language: tr
license: cc-by-sa-4.0
library_name: onnx
pipeline_tag: text-to-speech
tags: [turkish, tts, text-to-speech, styletts2, onnx, cpu, emotion]
---

# Seslen TTS — açık Türkçe ses modeli

> **English:** Turkish text-to-speech that runs on CPU with ONNX Runtime only (no PyTorch): 16 voices × 9 emotions (adult, young, elderly, child and cartoon voices),
> correct Turkish lexical stress and question intonation. StyleTTS2 student distilled from VoxCPM2-generated data
> (designed voices, not real people). Use with the [seslen-tts](https://github.com/seslen-tr/seslen-tts) Python package.

```python
# pip install "seslen-tts[hepsi]"
from seslen_tts import Seslen, wav_yaz
wav_yaz("merhaba.wav", Seslen().tum("Merhaba! Yarın bize gelecek misiniz?", ses="zeynep"))
```

## Dinleyin

Her cümlenin duygusu ayrı seçildi. Metin olduğu gibi verildi: saat, sayı ve kısaltmalar paketin içinde okunur hâle gelir.

**Ayşe** · olgun kadın · nötr → merak → nötr
> Hoş geldiniz, ben Ayşe. Size nasıl yardımcı olabilirim? Randevunuz yarın saat 10:30'da.

<audio controls src="https://huggingface.co/seslen-tr/seslen-tts/resolve/main/ornekler/ayse.wav"></audio>

**Zeynep** · genç kız · heyecan → neşeli → merak
> Harika bir haberim var! Sınavı geçtim! Akşam bize gelecek misin?

<audio controls src="https://huggingface.co/seslen-tr/seslen-tts/resolve/main/ornekler/zeynep.wav"></audio>

**Emre** · genç erkek · uyarı → nötr
> Dikkat edin, yol çok kaygan! Lütfen yavaş sürün.

<audio controls src="https://huggingface.co/seslen-tr/seslen-tts/resolve/main/ornekler/emre.wav"></audio>

**Fatma** · nine · masal üslubu (otomatik)
> Bir varmış, bir yokmuş. Evvel zaman içinde, küçük bir köyde meraklı bir kız yaşarmış.

<audio controls src="https://huggingface.co/seslen-tr/seslen-tts/resolve/main/ornekler/fatma.wav"></audio>

**Hasan** · dede · nötr
> Evladım, acele etme. Her işin bir vakti vardır.

<audio controls src="https://huggingface.co/seslen-tr/seslen-tts/resolve/main/ornekler/hasan.wav"></audio>

**Koca** · iri dev (çizgi karakter) · merak → kızgın
> Kim var orada? Benim ormanımda ne arıyorsun?

<audio controls src="https://huggingface.co/seslen-tr/seslen-tts/resolve/main/ornekler/koca.wav"></audio>

**Nehir** · genç kadın · ton otomatik
> Merhaba, ben Nehir. Toplantınız saat 14:00'te başlıyor; dosyaları sizin için hazırladım.

<audio controls src="https://huggingface.co/seslen-tr/seslen-tts/resolve/main/ornekler/nehir.wav"></audio>

**Ege** · genç erkek, derin ve sıcak · ton otomatik
> Merhaba, ben Ege. Bu akşam size sakin bir yolculuğun hikâyesini anlatacağım.

<audio controls src="https://huggingface.co/seslen-tr/seslen-tts/resolve/main/ornekler/ege.wav"></audio>

**Serdar** · olgun erkek · ton otomatik
> Arkadaşlar, toplantıya başlıyoruz. Lütfen telefonlarınızı sessize alın.

<audio controls src="https://huggingface.co/seslen-tr/seslen-tts/resolve/main/ornekler/serdar.wav"></audio>

**Selin** · genç kadın, yumuşak · ton otomatik
> Bugün kendine biraz zaman ayır. Derin bir nefes al ve rahatla.

<audio controls src="https://huggingface.co/seslen-tr/seslen-tts/resolve/main/ornekler/selin.wav"></audio>

**Elif** · kız çocuk · ton otomatik
> Anne, bak! Bahçede kocaman bir kelebek var! Yakalayabilir miyim?

<audio controls src="https://huggingface.co/seslen-tr/seslen-tts/resolve/main/ornekler/elif.wav"></audio>

**Can** · erkek çocuk · ton otomatik
> Bugün okulda resim yaptık. Benimki en güzeliydi, öğretmenim de çok beğendi!

<audio controls src="https://huggingface.co/seslen-tr/seslen-tts/resolve/main/ornekler/can.wav"></audio>

**Ece** · küçük kız · ton otomatik
> Bu oyuncak ayı benim en iyi arkadaşım. Adı Pamuk.

<audio controls src="https://huggingface.co/seslen-tr/seslen-tts/resolve/main/ornekler/ece.wav"></audio>

**Pıtırcık** · minik yaratık (çizgi karakter) · ton otomatik
> Yaşasın! Bugün ormanda kocaman bir şenlik var! Sen de gelecek misin?

<audio controls src="https://huggingface.co/seslen-tr/seslen-tts/resolve/main/ornekler/pitircik.wav"></audio>

**Fırfır** · kurnaz tilki (çizgi karakter) · ton otomatik
> Ay, ne kadar da güzel bir peynir o öyle. Bana biraz versene, dostum?

<audio controls src="https://huggingface.co/seslen-tr/seslen-tts/resolve/main/ornekler/firfir.wav"></audio>

**Kara** · kötü büyücü (çizgi karakter) · ton otomatik
> Sonunda büyülü kitap benim elimde! Artık kimse beni durduramaz!

<audio controls src="https://huggingface.co/seslen-tr/seslen-tts/resolve/main/ornekler/kara.wav"></audio>

## Dosyalar

| Dosya | İçerik |
|---|---|
| `metin.onnx` (78 MB) | metin kodlayıcı + Türkçe PL-BERT + süre tahmini: işaretler (1, L) + stil (1, 256) → d, t_en, süre |
| `ses.onnx` (247 MB) | çözücü (iSTFTNet): hizalanmış özellikler + stil + perde/tonlama + soru ezgisi → 24 kHz dalga |
| `stiller.npz` | 16 ses × 9 duygu stil vektörü (256 boyut) |
| `sesler.json`, `semboller.json` | ses bilgileri; okunuş karakteri → işaret numarası |
| `ornekler/*.wav` | her sesten bir örnek |

Paket ilk kullanımda yalnız ilk beş dosyayı indirir (`ornekler/` inmez). İnternetsiz kurulum için dosyaları bir
klasöre koyup `SESLEN_TTS_MODEL` ortam değişkeniyle ya da `Seslen(klasor="...")` ile gösterin.

## Sesler ve ölçümler

| Ses | Kim | UTMOS |
|---|---|---|
| nehir | genç kadın, net ve canlı | 3,08 |
| ege | genç erkek, derin ve sıcak | 3,82 |
| ayse | olgun kadın, güven veren | 4,01 |
| serdar | olgun erkek, otoriter ama dost | 3,46 |
| zeynep | genç kız, ince, canlı | 3,41 |
| emre | genç erkek, enerjik | 2,91 |
| selin | genç kadın, yumuşak | 3,06 |
| fatma | yaşlı kadın, şefkatli nine | 2,51 |
| hasan | yaşlı erkek, bilge dede | 3,47 |
| elif | kız çocuk (~8) | 2,21 |
| can | erkek çocuk (~9) | 1,29 |
| ece | küçük kız, çok ince | 1,86 |
| pitircik | minik sevimli yaratık (çizgi karakter) | 1,33 |
| koca | iri kalın dev (çizgi karakter) | 3,90 |
| firfir | kurnaz tilki (çizgi karakter) | 2,59 |
| kara | kötü büyücü (çizgi karakter) | 1,80 |

UTMOS yetişkin konuşmasıyla eğitildi; çocuk ve çizgi karakter seslerinde düşük puan kaliteden çok sesin
alışılmadık olmasını gösterir.

Duygular: nötr, neşeli, heyecan, uyarı, merak, özür, üzgün, kızgın, fısıltı.

- Okunuş denetimi (Türkçe wav2vec2 CTC, 16 ses × 14 cümle): 224 cümlede 6 sözcük hatası; dördü aynı ünlü ("başvurunuzu" → "başvurunuzı").
- Hız (i7-12700H, 6 iş parçacığı): gerçek zaman oranı ≈ 0,2–0,4.

## Eğitim

| Bileşen | Kaynak |
|---|---|
| Mimari | StyleTTS2 (iSTFTNet), LJSpeech ağırlığından başlatıldı |
| Ses ve duygu verisi | VoxCPM2 ile üretildi; sesler tariften tasarlandı, gerçek kişi değil; her kayıt otomatik okunuş denetiminden (Türkçe wav2vec2 CTC + sözcük başına GOP) geçti |
| Okunuş | turkce-vurgu (vurgu işaretli IPA) |
| Dil modeli | Türkçe PL-BERT ([seslen-tr/turkce-plbert](https://huggingface.co/seslen-tr/turkce-plbert)) |
| Donanım | tek RTX 4090; son ince ayar tek A100 |

## Sınırlılıklar

- **Ses klonlama yok.** Bilerek böyle: yalnız hazır sesler var.
- Yalnız Türkçe; yabancı sözcükler Türkçe okunuşla okunur.
- Duyguların belirginliği sese göre değişir.

## Lisans ve kullanım koşulları

- **Ağırlıklar ve sesler:** CC BY-SA 4.0 (Türkçe PL-BERT'in Vikipedi kaynaklı lisansı). Öğretmen VoxCPM2
  Apache-2.0; StyleTTS2 MIT. Paket kodu: MIT. Telaffuz için yardımcı veri: YODAS2 Türkçe kesitleri (Creative
  Commons lisanslı YouTube videoları, CC BY 3.0; Li vd. 2023, "YODAS: YouTube-Oriented Dataset for Audio and
  Speech"). Seslerin hiçbiri bu konuşmacılardan değildir; hepsi öğretmenin tarif tasarımıdır.
- **Kullanım koşulları:**
  - Üretilen sesin yapay zekâ sesi olduğu dinleyiciye belirtilmelidir.
  - Gerçek bir kişiyi taklit etmek, dolandırıcılık ve dezenformasyon için kullanmak yasaktır.
