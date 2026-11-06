package tr.seslen.deneme

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Python'daki seslen_tts'in ONNX oturumları (kopru.py _Oturum) buraya gelir: Android'de Python onnxruntime yok.
 * Girdi ve çıkışlar ham bayt (yerel bayt sırası) + şekil + tür (0 float32, 1 int64).
 */
class OrtKopru(private val xnnpack: Boolean = false) {
    private val ortam = OrtEnvironment.getEnvironment()
    private val oturumlar = HashMap<String, OrtSession>()

    /** xnnpack: ARM için optimize Conv / Gemm çekirdekleri (desteklemediği düğümler CPU'da kalır). */
    fun ac(ad: String, yol: String, isParcacigi: Int) {
        val ayar = OrtSession.SessionOptions()
        ayar.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        if (xnnpack) {
            // XNNPACK kendi iş parçacığı havuzunu kullanır; ORT'ninki 1 olmalı (yoksa ikisi çekirdek için yarışır)
            ayar.addXnnpack(mapOf("intra_op_num_threads" to isParcacigi.toString()))
            ayar.setIntraOpNumThreads(1)
            ayar.addConfigEntry("session.intra_op.allow_spinning", "0")
        } else {
            ayar.setIntraOpNumThreads(isParcacigi)
        }
        oturumlar.remove(ad)?.close()
        oturumlar[ad] = ortam.createSession(yol, ayar)
    }

    fun calistir(ad: String, adlar: Array<String>, sekiller: Array<LongArray>, turler: IntArray,
                 veriler: Array<ByteArray>): Sonuc {
        val oturum = oturumlar[ad] ?: error("oturum açılmadı: $ad")
        val girdiler = HashMap<String, OnnxTensor>()
        try {
            for (i in adlar.indices) {
                val b = ByteBuffer.allocateDirect(veriler[i].size).order(ByteOrder.nativeOrder())
                b.put(veriler[i]).rewind()
                girdiler[adlar[i]] = if (turler[i] == 0)
                    OnnxTensor.createTensor(ortam, b.asFloatBuffer(), sekiller[i])
                else
                    OnnxTensor.createTensor(ortam, b.asLongBuffer(), sekiller[i])
            }
            oturum.run(girdiler).use { r ->
                val cikis = ArrayList<Triple<Int, ByteArray, LongArray>>()
                for (i in 0 until r.size()) {
                    val t = r.get(i) as OnnxTensor
                    val sekil = t.info.shape
                    if (t.info.type == OnnxJavaType.INT64) {
                        val l = t.longBuffer
                        val b = ByteBuffer.allocate(l.remaining() * 8).order(ByteOrder.nativeOrder())
                        b.asLongBuffer().put(l)
                        cikis.add(Triple(1, b.array(), sekil))
                    } else {
                        val f = t.floatBuffer
                        val b = ByteBuffer.allocate(f.remaining() * 4).order(ByteOrder.nativeOrder())
                        b.asFloatBuffer().put(f)
                        cikis.add(Triple(0, b.array(), sekil))
                    }
                }
                return Sonuc(cikis)
            }
        } finally {
            girdiler.values.forEach { it.close() }
        }
    }

    class Sonuc(private val c: List<Triple<Int, ByteArray, LongArray>>) {
        fun sayi() = c.size
        fun tur(i: Int) = c[i].first
        fun veri(i: Int) = c[i].second
        fun sekil(i: Int) = c[i].third
    }
}
