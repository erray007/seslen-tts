package tr.seslen.deneme

import android.media.AudioFormat
import android.speech.tts.SynthesisCallback
import android.speech.tts.SynthesisRequest
import android.speech.tts.TextToSpeech
import android.speech.tts.TextToSpeechService
import android.speech.tts.Voice
import android.util.Log
import java.util.Locale

/**
 * Sistem TTS motoru: Ayarlar → Erişilebilirlik → Metin okuma çıkışı'nda "Seslen Deneme" görünür; ekran okuyucu ve
 * öteki uygulamalar kullanabilir. Yalnız Türkçe. Sesler "tur-TUR-x-<ad>" adlı 16 Voice; ses seçilmezse uygulamada
 * seçilen ses. Hız: istek hızı / 100 (0,5-2,5); perde: istek perdesi / 100 (0,7-1,4).
 */
class SeslenServisi : TextToSpeechService() {
    @Volatile private var durdu = false
    private val tr = Locale("tr", "TR")

    private fun turkce(dil: String?) = dil != null && (dil.equals("tur", true) || dil.equals("tr", true))

    override fun onIsLanguageAvailable(dil: String?, ulke: String?, varyant: String?): Int = when {
        !turkce(dil) -> TextToSpeech.LANG_NOT_SUPPORTED
        !Motor.indirildi(this) -> TextToSpeech.LANG_MISSING_DATA
        ulke.isNullOrEmpty() -> TextToSpeech.LANG_AVAILABLE
        ulke.equals("TUR", true) || ulke.equals("TR", true) -> TextToSpeech.LANG_COUNTRY_AVAILABLE
        else -> TextToSpeech.LANG_AVAILABLE
    }

    override fun onGetLanguage(): Array<String> = arrayOf("tur", "TUR", "")

    override fun onLoadLanguage(dil: String?, ulke: String?, varyant: String?): Int =
        onIsLanguageAvailable(dil, ulke, varyant)

    override fun onGetVoices(): List<Voice> = Motor.sesler(this).map { (ad, _, _) ->
        Voice(Motor.SES_ONEKI + ad, tr, Voice.QUALITY_VERY_HIGH, Voice.LATENCY_NORMAL, false, emptySet())
    }

    override fun onIsValidVoiceName(ad: String?): Int =
        if (ad != null && Motor.sesler(this).any { Motor.SES_ONEKI + it.first == ad }) TextToSpeech.SUCCESS
        else TextToSpeech.ERROR

    override fun onLoadVoice(ad: String?): Int = onIsValidVoiceName(ad)

    override fun onGetDefaultVoiceNameFor(dil: String?, ulke: String?, varyant: String?): String? =
        if (turkce(dil)) Motor.SES_ONEKI + Motor.varsayilanSes(this) else null

    override fun onCreate() {
        super.onCreate()
        // ısınma: ilk dokunuşta model açılışı ve ilk çalıştırma beklenmesin
        if (Motor.indirildi(this)) Thread {
            try { Motor.isit(this) } catch (e: Throwable) { Log.e("Seslen", "ısınma hatası", e) }
        }.start()
    }

    override fun onStop() {
        durdu = true
    }

    override fun onSynthesizeText(istek: SynthesisRequest, geri: SynthesisCallback) {
        durdu = false
        val metin = istek.charSequenceText?.toString() ?: istek.text ?: ""
        if (!Motor.indirildi(this) || !turkce(istek.language)) {
            geri.error(TextToSpeech.ERROR_NOT_INSTALLED_YET); return
        }
        val ses = istek.voiceName?.takeIf { it.startsWith(Motor.SES_ONEKI) }?.removePrefix(Motor.SES_ONEKI)
            ?: Motor.varsayilanSes(this)
        val hiz = (istek.speechRate / 100.0).coerceIn(0.5, 2.5)
        val perde = (istek.pitch / 100.0).coerceIn(0.7, 1.4)
        if (geri.start(Motor.ORAN, AudioFormat.ENCODING_PCM_16BIT, 1) != TextToSpeech.SUCCESS) return
        if (metin.isBlank()) { geri.done(); return }
        val enCok = geri.maxBufferSize
        try {
            var bekleyen = 0                                     // önceki cümleden sonra çalınacak sessizlik (örnek)
            Motor.konus(this, metin, ses, Motor.varsayilanTon(this), hiz, perde, { durdu }) { p ->
                if (bekleyen > 0) gonder(geri, ShortArray(bekleyen), enCok)
                val d = p.dalga
                val pcm = ShortArray(d.size) { i -> (d[i].coerceIn(-1f, 1f) * 32767f).toInt().toShort() }
                gonder(geri, pcm, enCok)
                bekleyen = Motor.ORAN * p.sessizlikMs / 1000
                Log.d("Seslen", "${p.cumle}: ${"%.0f".format(p.sureMs)} ms (${p.asama})")
            }
            if (durdu) geri.error(TextToSpeech.STOPPED) else geri.done()
        } catch (e: Throwable) {
            Log.e("Seslen", "üretim hatası", e)
            geri.error(TextToSpeech.ERROR_SYNTHESIS)
        }
    }

    private fun gonder(geri: SynthesisCallback, pcm: ShortArray, enCok: Int) {
        val bayt = ByteArray(pcm.size * 2)
        for (i in pcm.indices) {
            bayt[2 * i] = (pcm[i].toInt() and 0xff).toByte()
            bayt[2 * i + 1] = (pcm[i].toInt() shr 8).toByte()
        }
        var i = 0
        while (i < bayt.size && !durdu) {
            val n = minOf(enCok, bayt.size - i)
            geri.audioAvailable(bayt, i, n)
            i += n
        }
    }
}
