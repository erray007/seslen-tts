# Seslen Deneme (Android)

Telefonda Seslen'i denemek için: hem **sistem TTS motoru** (ekran okuyucu ve öteki uygulamalar kullanır) hem de
ölçüm ekranı (metin, ses, ton, hız → **Konuş**; ilk sesin süresi ve cümle başına RTF).

**Ekran okuyucuda:** Ayarlar → Erişilebilirlik → Metin okuma çıkışı → tercih edilen motor: **Seslen Deneme**
(uygulamadaki "Metin okuma ayarlarını aç" düğmesi oraya götürür). Ekran okuyucu, uygulamada seçilen sesi ve tonu
kullanır; 16 ses ayrıca `tur-TUR-x-<ad>` adlı Voice olarak listelenir. Hız ve perde ekran okuyucunun ayarından gelir.

- **Mantık değişmedi:** okunuş (turkce-vurgu + turkce-normalize), hizalama, soru ezgisi, duraklamalar, kısa cümle
  taşıyıcısı, hız dengesi deponun kökündeki `seslen_tts` paketinin kendisi; Chaquopy ile Python 3.11'de çalışır.
- **Yalnız ONNX oturumları Java'da:** Android'de Python onnxruntime yok; `app/src/main/python/kopru.py`
  oturumları `OrtKopru.kt`'ye (onnxruntime-android 1.29.0) yönlendirir.
- **Model APK'da değil:** ilk açılışta Hugging Face'ten (`seslen-tr/seslen-tts`) iner, ~325 MB.
- Yalnız arm64 (Chaquopy numpy). Duygu / üslup otomatiği yok (turkce-duygu, turkce-uslup eklenmedi): "otomatik"
  ton nötr okur.

## Derleme

```
# local.properties: sdk.dir=<Android SDK yolu>
# gradle.properties: seslen.python=<Python 3.11 yolu>   (Chaquopy paketleri onunla kurar)
./gradlew :app:assembleDebug        # → app/build/outputs/apk/debug/app-debug.apk (~39 MB)
```

Sürümler: AGP 8.5.2, Gradle 8.7, Kotlin 2.2.21, Chaquopy 16.1.0, compileSdk 35, minSdk 26.

## Düzen

| Dosya | İçerik |
|---|---|
| `app/build.gradle` | Chaquopy (Python 3.11, numpy, turkce-vurgu[tam]; `seslen_tts` kökten), onnxruntime-android |
| `app/src/main/python/kopru.py` | `seslen_tts.SesModeli`'nin Android hâli: ONNX oturumları Java'ya |
| `app/src/main/java/tr/seslen/deneme/OrtKopru.kt` | ONNX Runtime oturumları (bayt + şekil + tür ile) |
| `app/src/main/java/tr/seslen/deneme/ModelIndirici.kt` | ilk açılışta model indirme |
| `app/src/main/java/tr/seslen/deneme/Motor.kt` | uygulama ile servisin ortak motoru (tek Python, tek model), varsayılan ses / ton |
| `app/src/main/java/tr/seslen/deneme/SeslenServisi.kt` | sistem TTS motoru (TextToSpeechService): Türkçe, 16 Voice, hız / perde, durdurma |
| `app/src/main/java/tr/seslen/deneme/SesVerisi.kt` | sistemin ses verisi denetimi (CHECK_TTS_DATA) |
| `app/src/main/java/tr/seslen/deneme/MainActivity.kt` | model indirme, ses / ton seçimi, deneme ve ölçüm satırı |
| `app/src/main/res/xml/tts_motoru.xml` | TTS motoru tanımı (ayar ekranı) |

Bu ses yapay zekâ ile üretilir; uygulama bunu ekranda belirtir. Model: seslen-tts (Mehmet Veysel Delibaş), CC BY-SA 4.0.
