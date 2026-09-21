"""
Sarvam AI voice utilities for the Booth Agent.

Pipeline:
    microphone audio -> Sarvam Saaras STT -> Booth Agent -> Sarvam Bulbul TTS
"""

from __future__ import annotations

import base64
import os
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
    """Transcribe a Streamlit UploadedFile/file-like object with Sarvam."""
    client = get_client(api_key)

    # Streamlit UploadedFile is file-like. Rewind it so the SDK receives the
    # complete audio payload.
    try:
        audio_file.seek(0)
    except Exception:
        pass

    response = client.speech_to_text.transcribe(
        file=audio_file,
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
    return "data:audio/wav;base64," + base64.b64encode(audio_bytes).decode("ascii")
