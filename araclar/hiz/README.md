# Hız çalışması (06.11.2026)

Amaç: mevcut seslerle üretimi hızlandırmak ve ilk sesin gelme süresini kısaltmak (ekran okuyucu, Android).
Makine: 4 çekirdek Intel Xeon 2,3 GHz (AVX-512 VNNI), onnxruntime 1.29. Okunuş, turkce-vurgu kurulamadığı için
kaba harf çevirisi (`ortak.kaba`): zamanlama ve çözücü karşılaştırması için yeterli, dinleme için değil.

## Kurulum

```
pip install onnxruntime onnx numpy
export SESLEN_TTS_MODEL=/model/klasoru      # yoksa Hugging Face'ten iner
python araclar/hiz/girdiler.py              # cikti/cal_*.npz, cikti/test_*.npz
python araclar/hiz/yama.py sabit            # cikti/sabit.onnx (gürültüsüz karşılaştırma kopyası)
```

Ara dosyalar `araclar/hiz/cikti/` altında (git dışında).

## Bulgular

| | Sonuç |
|---|---|
| Temel (`olc.py`) | kısa 1,4 sn ses 0,32 sn (RTF 0,23), orta 3,8 sn 0,78 sn, uzun 6,3 sn 1,16 sn; ~%90 çözücüde |
| İş parçacığı | 1: RTF ~0,48; 2: ~0,30; 4: ~0,20 |
| Çözücü profili (`profil.py`) | Conv %57, Sin %19, InstanceNorm %6, ConvTranspose %5 |
| Sin (`mikro.py`) | ORT'de tek iş parçacığında; Mul 4 iş parçacığında 2,7 kat hızlanıyor |
| **int8, güvenli** (`nicele.py`) | resblocks 0-2 + noise_res: **×1,28, mel farkı 0,55 dB** (kıyas 0,52) |
| int8, + resblocks.3 | ×1,38, 0,63 dB; tanıyıcıyla 192 okumada birebir %83,9 (özgün %84,4); `model-int8` sürüm 1 |
| **int8, + resblocks 4-5 (convs2.2 hariç)** | 46 Conv, 0,85 dB, tanıyıcıyla %85,9 (özgün %85,9), ×1,42; **`model-int8` sürüm 2** |
| Duyarlılık (`nicele.py --tara`) | resblocks 4-5'te yalnız her bloğun son Conv'u (convs2.2) hassas: 0,81 / 0,76 dB, ötekiler 0,63-0,67 |
| Telefonda int8 (sürüm 1) | çözücü 785 → 536 ms (×1,46); nicelenen Conv ~310 → 61 ms (QLinearConv, ~5 kat); kulakla fark yok |
| Açılış | 0,97 sn; önceden eniyilenmiş .ort ile 0,74 sn (değmez) |
| `Seslen.dogal_hiz` | yeni (ses, ton) başına ilk cümleden önce ~190 ms (+ okunuş) |
| Kısa cümle taşıyıcısı (`tasiyici.py`) | 2 sözcüklük ifade 3 sözcüklükten 2 kat yavaş; yarım süreli taşıyıcı 674 → 479 ms, birebir %89 → %86 |
| Telefon (Xiaomi 17, 4 iş parçacığı) | RTF 0,47 (bu makine 0,20); %95 çözücüde |

## Elenenler

| Deneme | Sonuç | Neden |
|---|---|---|
| 1B Conv → 2B (H=1), NCHWc yolu | %9 yavaş | |
| ORT paralel yürütme (resblock dalları) | %50 yavaş | |
| Snake Sin → polinom (`yama.py poli`) | SNR 84 dB, %6 yavaş | 13 küçük işlem tek Sin'den pahalı |
| Dinamik int8 (Conv) | 7 kat yavaş | ConvInteger |
| Dinamik int8 (Gemm/MatMul) | hız yok, 1,18 dB | |
| Kısa cümlede taşıyıcıyı kırpmak (git geçmişinde) | 645 → 203 ms ama 6,9-8,3 dB | InstanceNorm bütün dalgaya bakıyor |
| Taşıyıcısız | 255 ms ama birebir %50 | uydurma hece geri geliyor |

## Sıradaki

1. ~~Ses önbelleği~~, ~~`dogal_hiz` önbelleği~~, ~~ısınma~~: Android 0.3'te.
2. Telefonda iş parçacığı sayısı ve XNNPACK karşılaştırması (Android 0.4).
3. İsteğe bağlı int8 çözücü; varsayılan olmadan önce kulakla A/B dinleme ve özdenetim (CTC) sınaması.
4. Snake için kaynaştırılmış özel çekirdek (çözücünün %19'u).
