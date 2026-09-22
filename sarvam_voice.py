"""
Sarvam AI voice utilities for Arjun / Booth Agent.

Pipeline:

    Streamlit microphone
        ->
    Sarvam Saaras STT
        ->
    Arjun
        ->
    Sarvam Bulbul TTS
"""

from __future__ import annotations

import base64

import requests
from sarvamai import SarvamAI


# ---------------------------------------------------------------------------
# Sarvam endpoints
# ---------------------------------------------------------------------------

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def get_client(api_key: str) -> SarvamAI:
    """Create a Sarvam SDK client."""

    if not api_key or not api_key.strip():
        raise ValueError(
            "SARVAM_API_KEY is not configured. "
            "Add it to Streamlit secrets or .env."
        )

    return SarvamAI(
        api_subscription_key=api_key.strip()
    )


# ---------------------------------------------------------------------------
# Speech to Text
# ---------------------------------------------------------------------------

def speech_to_text(
    audio_file,
    api_key: str,
    *,
    model: str = "saaras:v3",
    mode: str = "transcribe",
):
    """
    Transcribe a Streamlit UploadedFile using Sarvam's REST API.

    IMPORTANT:

    We intentionally do NOT pass the Streamlit UploadedFile directly
    into the Sarvam SDK.

    Streamlit Cloud can report browser microphone recordings with:

        audio/vnd.wave

    Sarvam accepts:

        audio/wav
        audio/x-wav
        audio/wave

    Therefore we read the bytes ourselves and explicitly construct
    the multipart request with:

        filename     = audio.wav
        Content-Type = audio/wav
    """

    if not api_key or not api_key.strip():
        raise ValueError(
            "SARVAM_API_KEY is not configured."
        )

    # -----------------------------------------------------------------------
    # Read the complete Streamlit upload
    # -----------------------------------------------------------------------

    try:
        audio_file.seek(0)
    except Exception:
        pass

    audio_bytes = audio_file.read()

    if not audio_bytes:
        raise ValueError(
            "The recorded audio file is empty."
        )

    # -----------------------------------------------------------------------
    # Diagnostic information
    # -----------------------------------------------------------------------

    original_name = getattr(
        audio_file,
        "name",
        None,
    )

    original_type = getattr(
        audio_file,
        "type",
        None,
    )

    print(
        "[Sarvam STT] "
        f"original_name={original_name!r}, "
        f"original_type={original_type!r}, "
        f"size={len(audio_bytes)} bytes"
    )

    # -----------------------------------------------------------------------
    # IMPORTANT:
    #
    # requests allows:
    #
    #     (filename, file_bytes, content_type)
    #
    # This forces the multipart part to be:
    #
    #     filename = audio.wav
    #     Content-Type = audio/wav
    #
    # Streamlit's original MIME type is therefore ignored.
    # -----------------------------------------------------------------------

    files = {
        "file": (
            "audio.wav",
            audio_bytes,
            "audio/wav",
        )
    }

    data = {
        "model": model,
        "mode": mode,
    }

    headers = {
        "api-subscription-key": api_key.strip(),
    }

    print(
        "[Sarvam STT] "
        "sending filename='audio.wav', "
        "content_type='audio/wav', "
        f"model={model!r}, "
        f"mode={mode!r}"
    )

    # -----------------------------------------------------------------------
    # Call Sarvam
    # -----------------------------------------------------------------------

    try:
        response = requests.post(
            SARVAM_STT_URL,
            headers=headers,
            data=data,
            files=files,
            timeout=60,
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not connect to Sarvam STT: {exc}"
        ) from exc

    # -----------------------------------------------------------------------
    # Handle HTTP errors
    # -----------------------------------------------------------------------

    if not response.ok:
        try:
            error_body = response.json()
        except Exception:
            error_body = response.text

        raise RuntimeError(
            "Sarvam STT request failed.\n"
            f"HTTP status: {response.status_code}\n"
            f"Response: {error_body}"
        )

    # -----------------------------------------------------------------------
    # Parse response
    # -----------------------------------------------------------------------

    try:
        result = response.json()
    except Exception as exc:
        raise RuntimeError(
            "Sarvam STT returned an invalid JSON response.\n"
            f"Response: {response.text[:1000]}"
        ) from exc

    print(
        "[Sarvam STT] "
        f"response keys={list(result.keys())}"
    )

    transcript = (
        result.get("transcript")
        or ""
    ).strip()

    language_code = (
        result.get("language_code")
        or "hi-IN"
    )

    return transcript, language_code


# ---------------------------------------------------------------------------
# Text to Speech
# ---------------------------------------------------------------------------

def text_to_speech(
    text: str,
    api_key: str,
    *,
    language_code: str = "hi-IN",
    model: str = "bulbul:v3",
    speaker: str = "shubh",
    pace: float = 1.0,
) -> bytes:
    """
    Synthesize text using Sarvam Bulbul.

    TTS continues to use the Sarvam Python SDK.
    """

    text = (text or "").strip()

    if not text:
        return b""

    # Bulbul v3 request limit.
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

    audios = getattr(
        response,
        "audios",
        None,
    ) or []

    if not audios:
        raise RuntimeError(
            "Sarvam TTS returned no audio."
        )

    audio = audios[0]

    if isinstance(audio, bytes):
        return audio

    return base64.b64decode(audio)


# ---------------------------------------------------------------------------
# Browser helper
# ---------------------------------------------------------------------------

def audio_bytes_to_data_url(
    audio_bytes: bytes,
) -> str:
    """
    Convert WAV bytes to a browser-playable data URL.
    """

    return (
        "data:audio/wav;base64,"
        + base64.b64encode(audio_bytes).decode("ascii")
    )