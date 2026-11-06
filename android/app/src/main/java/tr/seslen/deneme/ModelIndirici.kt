package tr.seslen.deneme

import java.io.File
import java.net.HttpURLConnection
import java.net.URL

/** Model dosyalarını ilk açılışta Hugging Face'ten indirir (~325 MB); yarım kalan dosya .part olarak kalır. */
object ModelIndirici {
    private const val DEPO = "https://huggingface.co/seslen-tr/seslen-tts/resolve/main/"
    val DOSYALAR = listOf("sesler.json", "semboller.json", "stiller.npz", "metin.onnx", "ses.onnx")

    fun hazir(klasor: File) = DOSYALAR.all { File(klasor, it).exists() }

    // INT8 ÇÖZÜCÜ (araclar/hiz/nicele.py): Hugging Face'te yok; çatalın model-int8 dalında 50 MB'lık parçalar
    // (GitHub dosya sınırı 100 MB). <ad>.json: {"parcalar": [...], "boyut": bayt, "sha256": "..."}
    const val INT8 = "ses_int8.onnx"
    private const val INT8_DEPO = "https://raw.githubusercontent.com/erray007/seslen-tts/model-int8/"

    private fun int8Bilgi() = org.json.JSONObject(URL(INT8_DEPO + "$INT8.json").readText())

    /** Daldaki int8 telefondakinden yeni mi (SHA-256 farklı)? Ağ yoksa false. */
    fun int8Yeni(klasor: File): Boolean = try {
        val dosya = File(klasor, INT8)
        val yerel = File(klasor, "$INT8.sha256")
        if (dosya.exists() && !yerel.exists()) yerel.writeText(sha256(dosya))     // 0.6'dan kalan: bir kez hesapla
        dosya.exists() && yerel.readText().trim() != int8Bilgi().getString("sha256")
    } catch (e: Throwable) { false }

    private fun sha256(f: File): String {
        val ozet = java.security.MessageDigest.getInstance("SHA-256")
        f.inputStream().use { g ->
            val t = ByteArray(1 shl 20)
            while (true) { val n = g.read(t); if (n < 0) break; ozet.update(t, 0, n) }
        }
        return ozet.digest().joinToString("") { "%02x".format(it) }
    }

    /** int8 çözücüyü parçalarından indirir, birleştirir, SHA-256 ile doğrular (guncelle: var olanın yerine yenisi).
     *  ilerleme(dosya, MB, toplam MB) */
    fun indirInt8(klasor: File, guncelle: Boolean = false, ilerleme: (String, Long, Long) -> Unit) {
        val hedef = File(klasor, INT8)
        if (hedef.exists() && !guncelle) return
        val bilgi = int8Bilgi()
        val parcalar = bilgi.getJSONArray("parcalar")
        val toplam = bilgi.getLong("boyut") shr 20
        val gecici = File(klasor, "$INT8.part")
        val ozet = java.security.MessageDigest.getInstance("SHA-256")
        var inen = 0L
        gecici.outputStream().use { c ->
            for (i in 0 until parcalar.length()) {
                val b = URL(INT8_DEPO + parcalar.getString(i)).openConnection() as HttpURLConnection
                b.connectTimeout = 30_000; b.readTimeout = 60_000
                if (b.responseCode !in 200..299) error("${parcalar.getString(i)} indirilemedi: HTTP ${b.responseCode}")
                b.inputStream.use { g ->
                    val tampon = ByteArray(1 shl 16)
                    var son = -1L
                    while (true) {
                        val n = g.read(tampon)
                        if (n < 0) break
                        c.write(tampon, 0, n); ozet.update(tampon, 0, n); inen += n
                        val mb = inen shr 20
                        if (mb != son) { son = mb; ilerleme(INT8, mb, toplam) }
                    }
                }
            }
        }
        val sha = ozet.digest().joinToString("") { "%02x".format(it) }
        if (sha != bilgi.getString("sha256")) { gecici.delete(); error("int8 çözücü bozuk indi (SHA-256 tutmadı)") }
        hedef.delete()
        if (!gecici.renameTo(hedef)) error("$INT8 kaydedilemedi")
        File(klasor, "$INT8.sha256").writeText(sha)
    }

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
