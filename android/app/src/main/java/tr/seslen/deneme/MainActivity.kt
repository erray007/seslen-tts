package tr.seslen.deneme

import android.app.Activity
import android.content.Intent
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.os.Bundle
import android.os.SystemClock
import android.text.InputType
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.Spinner
import android.widget.TextView
import java.util.concurrent.Executors
import java.util.concurrent.LinkedBlockingQueue
import kotlin.concurrent.thread

/**
 * Seslen deneme: model indirme, ses / ton seçimi (TTS motorunun varsayılanı olarak kaydedilir), Konuş ile deneme
 * ve ölçüm satırı (ilk sesin süresi, cümle başına RTF). Sistem TTS motoru: SeslenServisi.
 */
class MainActivity : Activity() {
    private val TONLAR = listOf("notr", "neseli", "heyecan", "uyari", "merak", "ozur", "uzgun", "kizgin", "fisilti")
    private val TON_ADLARI = listOf("Nötr", "Neşeli", "Heyecanlı", "Uyarı", "Meraklı", "Özür", "Üzgün", "Kızgın",
                                    "Fısıltı")
    private val isci = Executors.newSingleThreadExecutor()
    private lateinit var durum: TextView
    private lateinit var olcum: TextView
    private lateinit var metin: EditText
    private lateinit var ses: Spinner
    private lateinit var ton: Spinner
    private lateinit var hiz: SeekBar
    private lateinit var konus: Button
    private var sesler = listOf<String>()
    @Volatile private var durdur = false

    override fun onCreate(b: Bundle?) {
        super.onCreate(b)
        kur()
        isci.execute { hazirla() }
    }

    private fun kur() {
        val k = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(32, 32, 32, 32) }
        fun yazi(s: String) = TextView(this).apply { text = s; textSize = 18f }
        fun dugme(s: String, f: () -> Unit) = Button(this).apply { text = s; textSize = 20f; setOnClickListener { f() } }
        durum = yazi("Hazırlanıyor…").apply { accessibilityLiveRegion = View.ACCESSIBILITY_LIVE_REGION_POLITE }
        metin = EditText(this).apply {
            setText("Merhaba! Ben Seslen. Yarın saat 14:30'da toplantımız var, dosyaları hazırladın mı?")
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE
            minLines = 3; textSize = 18f; contentDescription = "Okunacak metin"
        }
        ses = Spinner(this).apply { contentDescription = "Ses" }
        ton = Spinner(this).apply {
            contentDescription = "Duygu tonu"
            adapter = ArrayAdapter(this@MainActivity, android.R.layout.simple_spinner_dropdown_item, TON_ADLARI)
            setSelection(maxOf(0, TONLAR.indexOf(Motor.varsayilanTon(this@MainActivity))))
            onItemSelectedListener = secildi { Motor.ayarlar(this@MainActivity).edit().putString("ton", TONLAR[it]).apply() }
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
        konus = dugme("Konuş") { basla() }.apply { isEnabled = false }
        val dur = dugme("Dur") { durdur = true }
        val ttsAyar = dugme("Metin okuma ayarlarını aç") {
            try { startActivity(Intent("com.android.settings.TTS_SETTINGS")) }
            catch (e: Throwable) { startActivity(Intent(android.provider.Settings.ACTION_ACCESSIBILITY_SETTINGS)) }
        }
        val bilgi = yazi("Ekran okuyucuda kullanmak için: Metin okuma ayarlarında tercih edilen motor olarak " +
                         "\"Seslen Deneme\"yi seçin. Ekran okuyucu bu ekranda seçilen sesi ve tonu kullanır.")
            .apply { textSize = 15f }
        olcum = yazi("").apply { setTextIsSelectable(true); textSize = 15f }
        val not = yazi("Bu ses yapay zekâ ile üretilmiştir. Model: seslen-tts (Mehmet Veysel Delibaş, CC BY-SA 4.0).")
            .apply { textSize = 13f }
        listOf(durum, metin, yazi("Ses"), ses, yazi("Ton"), ton, hizYazi, hiz, konus, dur, ttsAyar, bilgi, olcum, not)
            .forEach { k.addView(it) }
        setContentView(ScrollView(this).apply { addView(k) })
    }

    private fun secildi(f: (Int) -> Unit) = object : AdapterView.OnItemSelectedListener {
        override fun onItemSelected(p: AdapterView<*>?, v: View?, i: Int, id: Long) = f(i)
        override fun onNothingSelected(p: AdapterView<*>?) {}
    }

    private fun hizDegeri() = 0.5 + hiz.progress * 0.1
    private fun yaz(s: String) = runOnUiThread { durum.text = s }
    private fun ekle(s: String) = runOnUiThread { olcum.append(s + "\n") }

    private fun hazirla() {
        try {
            val klasor = Motor.klasor(this)
            if (!Motor.indirildi(this)) {
                yaz("Model indiriliyor (~325 MB, yalnız ilk açılışta)…")
                var son = -1L
                ModelIndirici.indir(klasor) { ad, mb, top ->
                    val y = if (top > 0) mb * 100 / top else 0
                    if (y / 10 != son / 10 || mb == top) { son = y; yaz("İndiriliyor: $ad, $mb / $top MB") }
                }
            }
            yaz("Model açılıyor…")
            Motor.ac(this)
            val liste = Motor.sesler(this)
            sesler = liste.map { it.first }
            runOnUiThread {
                ses.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item,
                                           liste.map { "${it.second}: ${it.third}" })
                ses.setSelection(maxOf(0, sesler.indexOf(Motor.varsayilanSes(this))))
                ses.onItemSelectedListener = secildi { Motor.ayarlar(this).edit().putString("ses", sesler[it]).apply() }
                konus.isEnabled = true
            }
            yaz("Hazır.")
            ekle("Açılış: %.2f sn, %d iş parçacığı".format(Motor.acilisSn, Motor.isParcacigi))
        } catch (e: Throwable) {
            yaz("Hata: ${e.message}")
        }
    }

    private fun basla() {
        val m = metin.text.toString()
        val s = sesler.getOrNull(ses.selectedItemPosition) ?: return
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
                Motor.konus(this, m, s, t, h, 1.0, { durdur }) { cumle, dalga, ms, tn, sure ->
                    val sn = dalga.size.toDouble() / Motor.ORAN
                    if (ilk) { ekle("İlk ses: ${SystemClock.elapsedRealtime() - t0} ms"); ilk = false }
                    ekle("• %s  [%s] %.2f sn ses, %.0f ms (RTF %.2f)".format(cumle, tn, sn, sure, sure / 1000 / sn))
                    topSes += sn; topSure += sure / 1000
                    kuyruk.put(dalga)
                    kuyruk.put(FloatArray(Motor.ORAN * ms / 1000))
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
        val tampon = AudioTrack.getMinBufferSize(Motor.ORAN, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_FLOAT)
        val iz = AudioTrack.Builder()
            .setAudioAttributes(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA)
                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
            .setAudioFormat(AudioFormat.Builder().setSampleRate(Motor.ORAN).setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
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
            if (durdur) { iz.pause(); iz.flush() } else iz.stop()
            iz.release()
        }
    }
}
