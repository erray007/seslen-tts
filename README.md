# seslen-tts int8 çözücü

`ses_int8.onnx`: [seslen-tts](https://huggingface.co/seslen-tr/seslen-tts) modelinin `ses.onnx` çözücüsünün
kısmen int8'e indirilmiş kopyası (Android deneme uygulaması bunu indirir). GitHub'ın 100 MB dosya sınırı yüzünden
50 MB'lık parçalar; `ses_int8.onnx.json` parça listesini, boyutu ve SHA-256'yı verir.

    cat ses_int8.onnx.p* > ses_int8.onnx

Nasıl üretildi (sürüm 2): `hiz-calismasi` dalında `araclar/hiz/nicele.py`; resblocks 0-3 ve noise_res'in bütün
Conv'ları, resblocks 4-5'in son Conv'u (convs2.2, nicelemeye hassas) dışındakiler: 46 Conv, QDQ, kanal başına int8
ağırlık, MinMax kalibrasyon; kaynak gürültüsü özgün gibi. Ölçü (06.11.2026): özdenetim tanıyıcısıyla 192 okumada
birebir %85,9 (özgün %85,9), log-mel farkı 0,85 dB (özgün ↔ özgün 0,52), işlemcide ~1,42 kat hızlı. Sürüm 1 (36 Conv,
%83,3, ×1,26) git geçmişinde değil: dal yeniden kuruldu.

**Lisans:** model ağırlıkları CC BY-SA 4.0. Özgün model: seslen-tts, Mehmet Veysel Delibaş (seslen-tr). Bu türev de
CC BY-SA 4.0'dır. Üretilen sesin yapay zekâ sesi olduğu dinleyiciye belirtilmelidir; gerçek bir kişiyi taklit etmek,
dolandırıcılık ve dezenformasyon için kullanmak yasaktır.
