package tr.seslen.deneme

import android.app.Activity
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.os.Bundle
import android.os.SystemClock
import android.text.InputType
import android.view.View
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.Spinner
import android.widget.TextView
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import org.json.JSONArray
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors
import java.util.concurrent.LinkedBlockingQueue
import kotlin.concurrent.thread

/**
 * Seslen deneme: metin, ses, ton, hız → Konuş. Cümleler hazır oldukça çalınır (bir cümle çalarken sonraki üretilir).
 * Ölçüm satırı: ilk sesin gelme süresi, her cümlenin üretim süresi ve gerçek zaman oranı (RTF).
 */
class MainActivity : Activity() {
    private val ORAN = 24000
    private val TONLAR = listOf("otomatik", "notr", "neseli", "heyecan", "uyari", "merak", "ozur", "uzgun", "kizgin",
                                "fisilti")
    private val isci = Executors.newSingleThreadExecutor()
    private lateinit var durum: TextView
    private lateinit var olcum: TextView
    private lateinit var metin: EditText
    private lateinit var ses: Spinner
    private lateinit var ton: Spinner
    private lateinit var hiz: SeekBar
    private lateinit var konus: Button
    private lateinit var dur: Button
    private var sesler = listOf<String>()
    private var kopru: PyObject? = null
    @Volatile private var durdur = false

    override fun onCreate(b: Bundle?) {
        super.onCreate(b)
        if (!Python.isStarted()) Python.start(AndroidPlatform(this))
        kur()
        isci.execute { hazirla() }
    }

    private fun kur() {
        val k = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(32, 32, 32, 32) }
        fun yazi(s: String) = TextView(this).apply { text = s; textSize = 18f }
        durum = yazi("Hazırlanıyor…").apply { accessibilityLiveRegion = View.ACCESSIBILITY_LIVE_REGION_POLITE }
        metin = EditText(this).apply {
            setText("Merhaba! Ben Seslen. Yarın saat 14:30'da toplantımız var, dosyaları hazırladın mı?")
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE
            minLines = 3; textSize = 18f; contentDescription = "Okunacak metin"
        }
        ses = Spinner(this).apply { contentDescription = "Ses" }
        ton = Spinner(this).apply {
            contentDescription = "Duygu tonu"
            adapter = ArrayAdapter(this@MainActivity, android.R.layout.simple_spinner_dropdown_item, TONLAR)
        }
        val hizYazi = yazi("Hız: 1,0")
        hiz = SeekBar(this).apply {
            max = 10; progress = 5; contentDescription = "Hız"
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(s: SeekBar?, p: Int, u: Boolean) {
                    hizYazi.text = "Hız: %.1f".format(hizDegeri()).replace('.', ',')
                    if (android.os.Build.VERSION.SDK_INT >= 30) stateDescription = hizYazi.text
                }
                override fun onStartTrackingTouch(s: SeekBar?) {}
                override fun onStopTrackingTouch(s: SeekBar?) {}
            })
        }
        konus = Button(this).apply { text = "Konuş"; textSize = 20f; isEnabled = false; setOnClickListener { basla() } }
        dur = Button(this).apply { text = "Dur"; textSize = 20f; setOnClickListener { durdur = true } }
        olcum = yazi("").apply { setTextIsSelectable(true); textSize = 15f }
        val not = yazi("Bu ses yapay zekâ ile üretilmiştir. Model: seslen-tts (Mehmet Veysel Delibaş, CC BY-SA 4.0).")
            .apply { textSize = 13f }
        listOf(durum, metin, yazi("Ses"), ses, yazi("Ton"), ton, hizYazi, hiz, konus, dur, olcum, not)
            .forEach { k.addView(it) }
        setContentView(ScrollView(this).apply { addView(k) })
    }

    private fun hizDegeri() = 0.5 + hiz.progress * 0.1

    private fun yaz(s: String) = runOnUiThread { durum.text = s }
    private fun ekle(s: String) = runOnUiThread { olcum.append(s + "\n") }

    private fun hazirla() {
        try {
            val klasor = File(filesDir, "model")
            if (!ModelIndirici.hazir(klasor)) {
                yaz("Model indiriliyor (~325 MB, yalnız ilk açılışta)…")
                var sonYuzde = -1L
                ModelIndirici.indir(klasor) { ad, mb, top ->
                    val y = if (top > 0) mb * 100 / top else 0
                    if (y / 10 != sonYuzde / 10 || mb == top) {
                        sonYuzde = y; yaz("İndiriliyor: $ad, $mb / $top MB")
                    }
                }
            }
            yaz("Model açılıyor…")
            val py = Python.getInstance().getModule("kopru")
            val isParcacigi = minOf(4, Runtime.getRuntime().availableProcessors())
            val sure = py.callAttr("hazirla", klasor.absolutePath, OrtKopru(), isParcacigi).toDouble()
            val liste = JSONArray(py.callAttr("sesler").toString())
            val adlar = ArrayList<String>(); val gorunen = ArrayList<String>()
            for (i in 0 until liste.length()) {
                val s = liste.getJSONArray(i); adlar.add(s.getString(0)); gorunen.add("${s.getString(1)}: ${s.getString(2)}")
            }
            kopru = py; sesler = adlar
            runOnUiThread {
                ses.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, gorunen)
                konus.isEnabled = true
            }
            yaz("Hazır."); ekle("Açılış: %.2f sn, %d iş parçacığı".format(sure, isParcacigi))
        } catch (e: Throwable) {
            yaz("Hata: ${e.message}")
        }
    }

    private fun basla() {
        val py = kopru ?: return
        val m = metin.text.toString()
        val s = sesler[ses.selectedItemPosition]
        val t = TONLAR[ton.selectedItemPosition]
        val h = hizDegeri()
        durdur = false
        konus.isEnabled = false
        yaz("Konuşuyor…")
        isci.execute {
            val kuyruk = LinkedBlockingQueue<FloatArray>()
            val calici = thread { cal(kuyruk) }
            val t0 = SystemClock.elapsedRealtime()
            var ilk = true
            var topSes = 0.0; var topSure = 0.0
            try {
                val gen = py.callAttr("seslendir", m, s, t, h)
                val sonraki = Python.getInstance().builtins["next"]!!
                while (!durdur) {
                    val p = sonraki.call(gen, null) ?: break
                    val l = p.asList()
                    val bayt = l[1].toJava(ByteArray::class.java)
                    val dalga = FloatArray(bayt.size / 4)
                    ByteBuffer.wrap(bayt).order(ByteOrder.nativeOrder()).asFloatBuffer().get(dalga)
                    val sn = dalga.size.toDouble() / ORAN
                    val ms = l[4].toDouble()
                    if (ilk) { ekle("İlk ses: ${SystemClock.elapsedRealtime() - t0} ms"); ilk = false }
                    ekle("• %s  [%s] %.2f sn ses, %.0f ms (RTF %.2f)".format(l[0].toString(), l[3].toString(), sn, ms,
                                                                             ms / 1000 / sn))
                    topSes += sn; topSure += ms / 1000
                    kuyruk.put(dalga)
                    kuyruk.put(FloatArray(ORAN * l[2].toInt() / 1000))
                }
                if (topSes > 0) ekle("Toplam: %.1f sn ses, %.1f sn üretim, RTF %.2f\n".format(topSes, topSure, topSure / topSes))
            } catch (e: Throwable) {
                ekle("Hata: ${e.message}")
            } finally {
                kuyruk.put(FloatArray(0))
                calici.join()
                runOnUiThread { konus.isEnabled = true }
                yaz(if (durdur) "Durduruldu." else "Bitti.")
            }
        }
    }

    /** Kuyruktaki dalgaları çalar; boş dizi bitiş işaretidir. */
    private fun cal(kuyruk: LinkedBlockingQueue<FloatArray>) {
        val tampon = AudioTrack.getMinBufferSize(ORAN, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_FLOAT)
        val iz = AudioTrack.Builder()
            .setAudioAttributes(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA)
                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
            .setAudioFormat(AudioFormat.Builder().setSampleRate(ORAN).setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                .setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
            .setBufferSizeInBytes(tampon * 2).setTransferMode(AudioTrack.MODE_STREAM).build()
        iz.play()
        try {
            while (true) {
                val d = kuyruk.take()
                if (d.isEmpty() || durdur) break
                var i = 0
                while (i < d.size && !durdur) {
                    val n = iz.write(d, i, minOf(4096, d.size - i), AudioTrack.WRITE_BLOCKING)
                    if (n <= 0) break
                    i += n
                }
            }
        } finally {
            if (durdur) iz.pause().also { iz.flush() } else iz.stop()
            iz.release()
        }
    }
}
