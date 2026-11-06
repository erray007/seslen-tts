package tr.seslen.deneme

import java.io.File
import java.net.HttpURLConnection
import java.net.URL

/** Model dosyalarını ilk açılışta Hugging Face'ten indirir (~325 MB); yarım kalan dosya .part olarak kalır. */
object ModelIndirici {
    private const val DEPO = "https://huggingface.co/seslen-tr/seslen-tts/resolve/main/"
    val DOSYALAR = listOf("sesler.json", "semboller.json", "stiller.npz", "metin.onnx", "ses.onnx")

    fun hazir(klasor: File) = DOSYALAR.all { File(klasor, it).exists() }

    /** ilerleme(dosya, indirilen MB, toplam MB) */
    fun indir(klasor: File, ilerleme: (String, Long, Long) -> Unit) {
        klasor.mkdirs()
        for (ad in DOSYALAR) {
            val hedef = File(klasor, ad)
            if (hedef.exists()) continue
            val gecici = File(klasor, "$ad.part")
            val b = URL(DEPO + ad).openConnection() as HttpURLConnection
            b.instanceFollowRedirects = true
            b.connectTimeout = 30_000
            b.readTimeout = 60_000
            if (b.responseCode !in 200..299) error("$ad indirilemedi: HTTP ${b.responseCode}")
            val toplam = b.contentLengthLong
            var inen = 0L
            var son = -1L
            b.inputStream.use { g ->
                gecici.outputStream().use { c ->
                    val tampon = ByteArray(1 shl 16)
                    while (true) {
                        val n = g.read(tampon)
                        if (n < 0) break
                        c.write(tampon, 0, n)
                        inen += n
                        val mb = inen shr 20
                        if (mb != son) { son = mb; ilerleme(ad, mb, toplam shr 20) }
                    }
                }
            }
            if (!gecici.renameTo(hedef)) error("$ad kaydedilemedi")
        }
    }
}
