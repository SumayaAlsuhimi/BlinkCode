from gtts import gTTS
from io import BytesIO


def text_to_speech(text, lang="ar"):
    if not text or not text.strip():
        return None

    audio_buffer = BytesIO()

    tts = gTTS(
        text=text,
        lang=lang
    )

    tts.write_to_fp(audio_buffer)

    audio_buffer.seek(0)

    return audio_buffer