package tr.seslen.deneme

import android.content.Context
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Uygulama ile TTS servisinin ortak motoru (aynı süreç, tek Python, tek model). İlk kullanımda açılır.
 */
object Motor {
    const val ORAN = 24000
    const val SES_ONEKI = "tur-TUR-x-"
    private val kilit = Any()
    @Volatile private var py: PyObject? = null
    var acilisSn = 0.0
        private set
    var isParcacigi = 4
        private set
    var xnnpack = false
        private set

    fun klasor(c: Context) = File(c.filesDir, "model")
    fun indirildi(c: Context) = ModelIndirici.hazir(klasor(c))
    fun acik() = py != null

    /** [ad, görünen ad, açıklama]: sesler.json'dan, Python açılmadan (servis ses listesini erken ister). */
    fun sesler(c: Context): List<Triple<String, String, String>> {
        val f = File(klasor(c), "sesler.json")
        if (!f.exists()) return emptyList()
        val j = JSONObject(f.readText())
        return j.keys().asSequence().map { a ->
            val s = j.getJSONObject(a); Triple(a, s.optString("gorunen", a), s.optString("aciklama", ""))
        }.toList()
    }

    fun ayarlar(c: Context) = c.getSharedPreferences("seslen", Context.MODE_PRIVATE)
    fun varsayilanSes(c: Context) = ayarlar(c).getString("ses", "nehir")!!
    fun varsayilanTon(c: Context) = ayarlar(c).getString("ton", "notr")!!

    fun ac(c: Context): PyObject = py ?: synchronized(kilit) {
        py ?: run {
            if (!Python.isStarted()) Python.start(AndroidPlatform(c.applicationContext))
            val m = Python.getInstance().getModule("kopru")
            isParcacigi = istenenIsParcacigi(c)
            xnnpack = false      // XNNPACK: Xiaomi 17'de RTF 1,57 (işlemci 0,47); seçenek kaldırıldı, kod OrtKopru'da
            acilisSn = m.callAttr("hazirla", klasor(c).absolutePath, OrtKopru(xnnpack), isParcacigi, cozucu(c)).toDouble()
            m.callAttr("ayarla", tasiyiciKat(c))
            py = m
            m
        }
    }

    /** Ayardaki iş parçacığı sayısı (yoksa en çok 4). Telefonda büyük / küçük çekirdek karışık: en iyisi ölçülür. */
    fun istenenIsParcacigi(c: Context) =
        ayarlar(c).getInt("is_parcacigi", minOf(4, Runtime.getRuntime().availableProcessors()))

    /**
     * Telefonda profil: oturumlar profilleme açık yeniden açılır, örnek ifade 3 kez (önbelleksiz) üretilir, işlem
     * türlerine göre süreler döner; sonra oturumlar normal açılır.
     */
    fun profilCikar(c: Context, metin: String): String = synchronized(kilit) {
        val m = ac(c)
        val kopru = OrtKopru(xnnpack, c.cacheDir.absolutePath)
        m.callAttr("hazirla", klasor(c).absolutePath, kopru, isParcacigi, cozucu(c))
        m.callAttr("ayarla", tasiyiciKat(c))
        val ses = varsayilanSes(c)
        val gen = Python.getInstance().builtins["list"]!!
        repeat(3) { gen.call(m.callAttr("seslendir", metin, ses, "notr", 1.0, 1.0)) }
        val ozet = kopru.profilOzeti("ses") + kopru.profilOzeti("metin")
        acilisSn = m.callAttr("hazirla", klasor(c).absolutePath, OrtKopru(xnnpack), isParcacigi, cozucu(c)).toDouble()
        isinan.clear()
        "Profil ($isParcacigi iş parçacığı, ${cozucu(c)}):\n$ozet"
    }

    /** Kullanılan çözücü dosyası: ses.onnx ya da int8 kopyası (indirildiyse). */
    fun cozucu(c: Context): String {
        val d = ayarlar(c).getString("cozucu", "ses.onnx")!!
        return if (File(klasor(c), d).exists()) d else "ses.onnx"
    }

    /** Çözücü değişti: kaydedilir, oturumlar yeniden açılır. Süre sn. */
    fun cozucuSec(c: Context, dosya: String): Double {
        ayarlar(c).edit().putString("cozucu", dosya).apply()
        return yenidenAc(c, isParcacigi, xnnpack)
    }

    /** Kısa ifade taşıyıcısının süre katı (1 olduğu gibi, 0,5 hızlı). */
    fun tasiyiciKat(c: Context) = ayarlar(c).getFloat("tasiyici_kat", 1f).toDouble()

    fun tasiyiciAyarla(c: Context, kat: Double) {
        ayarlar(c).edit().putFloat("tasiyici_kat", kat.toFloat()).apply()
        py?.callAttr("ayarla", kat)
        synchronized(onbellek) { onbellek.clear(); onbellekOrnek = 0 }
    }

    /** İş parçacığı ya da hızlandırıcı değişti: oturumlar yeniden açılır (önbellek ve ısınma sıfırlanır). Süre sn. */
    fun yenidenAc(c: Context, n: Int, xnn: Boolean): Double = synchronized(kilit) {
        ayarlar(c).edit().putInt("is_parcacigi", n).putBoolean("xnnpack", xnn).apply()
        val m = py ?: return ac(c).let { acilisSn }
        isParcacigi = n; xnnpack = xnn
        acilisSn = m.callAttr("hazirla", klasor(c).absolutePath, OrtKopru(xnn), n, cozucu(c)).toDouble()
        isinan.clear()
        synchronized(onbellek) { onbellek.clear(); onbellekOrnek = 0 }
        acilisSn
    }

    /** Sesin doğal hızı, harf adları ve ONNX Runtime'ın ilk (yavaş) çalıştırması: ilk istekten önce. Süre sn. */
    fun isit(c: Context, ses: String = varsayilanSes(c), ton: String = varsayilanTon(c)): Double {
        val m = ac(c)
        val anahtar = "$ses/$ton"
        if (anahtar in isinan) return 0.0
        val sn = m.callAttr("isit", ses, ton).toDouble()
        isinan.add(anahtar)
        return sn
    }
    private val isinan = java.util.Collections.synchronizedSet(HashSet<String>())

    /** Bir cümlenin sonucu: önbellekte de bu tutulur. */
    class Parca(val cumle: String, val dalga: FloatArray, val sessizlikMs: Int, val ton: String,
                val sureMs: Double, val asama: String)

    // SES ÖNBELLEĞİ: ekran okuyucu aynı ifadeleri sık tekrarlıyor ("Geri düğmesi", "Ana ekran"). Anahtar metin + ses +
    // ton + hız + perde; en son kullanılan kalır, toplam ONBELLEK_ORNEK örnek (float, ~60 sn ses, ~6 MB).
    private const val ONBELLEK_ORNEK = 60 * ORAN
    private const val ONBELLEK_EN_UZUN = 10 * ORAN          // bundan uzun metin önbelleğe girmez
    private val onbellek = object : LinkedHashMap<String, List<Parca>>(64, 0.75f, true) {}
    private var onbellekOrnek = 0

    private fun onbellekAl(k: String) = synchronized(onbellek) { onbellek[k] }
    private fun onbellegeKoy(k: String, p: List<Parca>) = synchronized(onbellek) {
        val n = p.sumOf { it.dalga.size }
        if (n > ONBELLEK_EN_UZUN || k in onbellek) return
        onbellek[k] = p; onbellekOrnek += n
        val i = onbellek.entries.iterator()
        while (onbellekOrnek > ONBELLEK_ORNEK && i.hasNext()) {
            val e = i.next(); onbellekOrnek -= e.value.sumOf { it.dalga.size }; i.remove()
        }
    }

    /**
     * Metni cümle cümle üretir; her cümlede parca(...) çağrılır (önbellekteyse hemen, sureMs 0).
     * durdu() true dönerse sonraki cümleye geçilmez.
     */
    fun konus(c: Context, metin: String, ses: String, ton: String, hiz: Double, perde: Double,
              durdu: () -> Boolean, parca: (Parca) -> Unit) {
        val k = "$ses|$ton|%.2f|%.2f|$metin".format(hiz, perde)
        onbellekAl(k)?.let { liste ->
            for (p in liste) { if (durdu()) return; parca(Parca(p.cumle, p.dalga, p.sessizlikMs, p.ton, 0.0, "önbellek")) }
            return
        }
        val m = ac(c)
        val gen = m.callAttr("seslendir", metin, ses, ton, hiz, perde)
        val sonraki = Python.getInstance().builtins["next"]!!
        val hepsi = ArrayList<Parca>()
        var tamam = false
        try {
            while (!durdu()) {
                val p = sonraki.call(gen, null)
                if (p == null) { tamam = true; break }
                val l = p.asList()
                val bayt = l[1].toJava(ByteArray::class.java)
                val dalga = FloatArray(bayt.size / 4)
                ByteBuffer.wrap(bayt).order(ByteOrder.nativeOrder()).asFloatBuffer().get(dalga)
                val yeni = Parca(l[0].toString(), dalga, l[2].toInt(), l[3].toString(), l[4].toDouble(), l[5].toString())
                hepsi.add(yeni)
                parca(yeni)
            }
        } finally {
            gen.callAttr("close")
        }
        if (tamam) onbellegeKoy(k, hepsi)
    }
}
