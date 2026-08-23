import asyncio
import sounddevice as sd
from app import decode_webm_to_pcm

VOICE_MAP = {
    "en-IN": "en-IN-NeerjaNeural",
    "en-US": "en-US-AriaNeural",
    "en-GB": "en-GB-SoniaNeural",
    "hi-IN": "hi-IN-SwaraNeural",
    "es-ES": "es-ES-ElviraNeural"
}

def speak_text_sync(text: str, lang_code: str = "en-IN", sample_rate: int = 24000, visualizer_callback=None):
    try:
        import edge_tts
        voice = VOICE_MAP.get(lang_code, "en-IN-NeerjaNeural")
        
        async def _synth():
            communicate = edge_tts.Communicate(text, voice)
            audio_bytes = b''
            async for chunk in communicate.stream():
                if chunk['type'] == 'audio':
                    audio_bytes += chunk['data']
            return audio_bytes

        raw_audio = asyncio.run(_synth())
        if not raw_audio:
            return
            
        pcm = decode_webm_to_pcm(raw_audio, target_sr=sample_rate)
        sd.play(pcm, samplerate=sample_rate)
        
        if visualizer_callback:
            chunk_len = int(sample_rate * 0.05)
            for i in range(0, len(pcm), chunk_len):
                if not sd.get_stream().active:
                    break
                sub_chunk = pcm[i:i+chunk_len]
                visualizer_callback(sub_chunk)
                sd.sleep(50)
        else:
            sd.wait()
            
    except Exception:
        try:
            from gtts import gTTS
            import io
            g_lang = "hi" if "hi" in lang_code else "es" if "es" in lang_code else "en"
            tts = gTTS(text=text, lang=g_lang, slow=False)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            pcm = decode_webm_to_pcm(fp.read(), target_sr=sample_rate)
            sd.play(pcm, samplerate=sample_rate)
            sd.wait()
        except Exception:
            pass
