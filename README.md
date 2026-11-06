# seslen-tts int8 çözücü

`ses_int8.onnx`: [seslen-tts](https://huggingface.co/seslen-tr/seslen-tts) modelinin `ses.onnx` çözücüsünün
kısmen int8'e indirilmiş kopyası (Android deneme uygulaması bunu indirir). GitHub'ın 100 MB dosya sınırı yüzünden
50 MB'lık parçalar; `ses_int8.onnx.json` parça listesini, boyutu ve SHA-256'yı verir.

    cat ses_int8.onnx.p* > ses_int8.onnx

Nasıl üretildi: `hiz-calismasi` dalında `araclar/hiz/nicele.py resblocks.0,resblocks.1,resblocks.2,resblocks.3,noise_res
cikti/d_g3.onnx --gurultu` (QDQ, kanal başına int8 ağırlık, MinMax kalibrasyon, 36 Conv; kaynak gürültüsü özgün gibi).
Ölçü (06.11.2026): log-mel farkı 0,63 dB (özgün ↔ özgün 0,52); özdenetim tanıyıcısıyla 192 okumada birebir %83,9
(özgün %84,4); işlemcide ~1,26 kat hızlı.

**Lisans:** model ağırlıkları CC BY-SA 4.0. Özgün model: seslen-tts, Mehmet Veysel Delibaş (seslen-tr). Bu türev de
CC BY-SA 4.0'dır. Üretilen sesin yapay zekâ sesi olduğu dinleyiciye belirtilmelidir; gerçek bir kişiyi taklit etmek,
dolandırıcılık ve dezenformasyon için kullanmak yasaktır.
