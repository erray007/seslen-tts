---
language: tr
license: cc-by-sa-4.0
library_name: onnx
pipeline_tag: text-to-speech
tags: [turkish, tts, text-to-speech, styletts2, onnx, cpu, emotion]
---

# Seslen TTS — açık Türkçe ses modeli

> **English:** Turkish text-to-speech that runs on CPU with ONNX Runtime only (no PyTorch): 6 voices × 9 emotions,
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

## Dosyalar

| Dosya | İçerik |
|---|---|
| `metin.onnx` (78 MB) | metin kodlayıcı + Türkçe PL-BERT + süre tahmini: işaretler (1, L) + stil (1, 256) → d, t_en, süre |
| `ses.onnx` (247 MB) | çözücü (iSTFTNet): hizalanmış özellikler + stil + perde/tonlama + soru ezgisi → 24 kHz dalga |
| `stiller.npz` | 6 ses × 9 duygu stil vektörü (256 boyut) |
| `sesler.json`, `semboller.json` | ses bilgileri; okunuş karakteri → işaret numarası |
| `ornekler/*.wav` | her sesten bir örnek |

Paket ilk kullanımda yalnız ilk beş dosyayı indirir (`ornekler/` inmez). İnternetsiz kurulum için dosyaları bir
klasöre koyup `SESLEN_TTS_MODEL` ortam değişkeniyle ya da `Seslen(klasor="...")` ile gösterin.

## Sesler ve ölçümler

| Ses | Kim | UTMOS |
|---|---|---|
| ayse | olgun kadın, güven veren | 3,89 |
| koca | iri kalın dev (çizgi karakter) | 3,89 |
| hasan | yaşlı erkek, bilge dede | 3,26 |
| zeynep | genç kız, ince, canlı | 3,25 |
| emre | genç erkek, enerjik | 2,87 |
| fatma | yaşlı kadın, şefkatli nine | 2,38 |

Duygular: nötr, neşeli, heyecan, uyarı, merak, özür, üzgün, kızgın, fısıltı.

- Okunuş denetimi (Türkçe wav2vec2 CTC, 6 ses × 14 cümle): 84 cümlede 1 sözcük hatası (Hasan: "aradığın" → "aradığım").
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
