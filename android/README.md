# Seslen Deneme (Android)

Telefonda hız ve ilk sesin gelme süresini görmek için küçük deneme uygulaması: metin, ses, ton, hız → **Konuş**.
Cümleler hazır oldukça çalınır; her cümlenin üretim süresi ve gerçek zaman oranı (RTF) ekrana yazılır.

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
| `app/src/main/java/tr/seslen/deneme/MainActivity.kt` | arayüz, cümle cümle üretim ve çalma, ölçüm satırı |

Bu ses yapay zekâ ile üretilir; uygulama bunu ekranda belirtir. Model: seslen-tts (Mehmet Veysel Delibaş), CC BY-SA 4.0.
