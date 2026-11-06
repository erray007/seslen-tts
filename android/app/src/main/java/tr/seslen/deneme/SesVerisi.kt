package tr.seslen.deneme

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.speech.tts.TextToSpeech

/** Sistemin "ses verisi var mı?" sorusu (CHECK_TTS_DATA): model indirildiyse Türkçe hazır. */
class SesVerisi : Activity() {
    override fun onCreate(b: Bundle?) {
        super.onCreate(b)
        val hazir = Motor.indirildi(this)
        setResult(
            if (hazir) TextToSpeech.Engine.CHECK_VOICE_DATA_PASS else TextToSpeech.Engine.CHECK_VOICE_DATA_FAIL,
            Intent()
                .putStringArrayListExtra(TextToSpeech.Engine.EXTRA_AVAILABLE_VOICES,
                                         if (hazir) arrayListOf("tur-TUR") else arrayListOf())
                .putStringArrayListExtra(TextToSpeech.Engine.EXTRA_UNAVAILABLE_VOICES,
                                         if (hazir) arrayListOf() else arrayListOf("tur-TUR")))
        finish()
    }
}
