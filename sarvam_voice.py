"""
Sarvam AI voice utilities for the Booth Agent.

Pipeline:
    microphone audio -> Sarvam Saaras STT -> Booth Agent -> Sarvam Bulbul TTS
"""

from __future__ import annotations

import base64
from io import BytesIO

from sarvamai import SarvamAI


def get_client(api_key: str) -> SarvamAI:
    if not api_key:
        raise ValueError(
            "SARVAM_API_KEY is not configured. Add it to .env or Streamlit secrets."
        )
    return SarvamAI(api_subscription_key=api_key)


def speech_to_text(
    audio_file,
    api_key: str,
    *,
    model: str = "saaras:v3",
    mode: str = "transcribe",
):
    """
    Transcribe a Streamlit UploadedFile/file-like object with Sarvam.

    Streamlit/browser audio can sometimes report the MIME type as
    `audio/vnd.wave`, which Sarvam rejects even though it accepts
    `audio/wav`.

    We therefore normalize the MIME type and send the audio as a
    standard WAV file.
    """

    client = get_client(api_key)

    # Read the complete uploaded audio payload.
    try:
        audio_file.seek(0)
    except Exception:
        pass

    audio_bytes = audio_file.read()

    if not audio_bytes:
        raise ValueError("No audio data was received.")

    # Always send a normal WAV MIME type to Sarvam.
    #
    # The important part of this fix is that we do NOT pass the original
    # Streamlit UploadedFile directly, because its MIME type can be:
    #
    #     audio/vnd.wave
    #
    # Sarvam rejects that value.
    #
    # BytesIO alone does not necessarily give the SDK the MIME type we want,
    # so create a file-like object with a standard `.name`.
    normalized_audio = BytesIO(audio_bytes)
    normalized_audio.name = "audio.wav"

    response = client.speech_to_text.transcribe(
        file=normalized_audio,
        model=model,
        mode=mode,
    )

    transcript = getattr(response, "transcript", "") or ""
    language_code = getattr(response, "language_code", None) or "hi-IN"

    return transcript.strip(), language_code


def text_to_speech(
    text: str,
    api_key: str,
    *,
    language_code: str = "hi-IN",
    model: str = "bulbul:v3",
    speaker: str = "shubh",
    pace: float = 1.0,
) -> bytes:
    """Synthesize text with Sarvam Bulbul and return WAV bytes."""

    text = (text or "").strip()

    if not text:
        return b""

    # Bulbul v3 accepts up to 2500 characters per REST request.
    if len(text) > 2500:
        text = text[:2497].rstrip() + "..."

    client = get_client(api_key)

    response = client.text_to_speech.convert(
        text=text,
        language_code=language_code,
        model=model,
        speaker=speaker,
        pace=pace,
        output_audio_codec="wav",
        speech_sample_rate=24000,
    )

    audios = getattr(response, "audios", None) or []

    if not audios:
        raise RuntimeError("Sarvam TTS returned no audio.")

    audio = audios[0]

    if isinstance(audio, bytes):
        return audio

    return base64.b64decode(audio)


def audio_bytes_to_data_url(audio_bytes: bytes) -> str:
    """Convert audio bytes to a browser-playable data URL if needed."""

    return (
        "data:audio/wav;base64,"
        + base64.b64encode(audio_bytes).decode("ascii")
    )