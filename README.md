# booth-agent


## Voice mode — Sarvam AI

Arjun now supports a voice conversation loop:

`Microphone → Sarvam Saaras STT → Arjun → Sarvam Bulbul TTS → Browser audio`

### Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Add your Sarvam API key to `.env`:

```env
SARVAM_API_KEY=YOUR_SARVAM_API_KEY
SARVAM_STT_MODEL=saaras:v3
SARVAM_STT_MODE=transcribe
SARVAM_TTS_MODEL=bulbul:v3
SARVAM_TTS_SPEAKER=shubh
SARVAM_TTS_PACE=1.0
SARVAM_TTS_LANGUAGE=hi-IN
```

3. Start Streamlit:

```bash
streamlit run streamlit_app.py
```

Use **Talk to Arjun** to record a question. The transcript is sent through the existing Arjun agent, and the response is spoken back using Sarvam TTS.

Typed chat continues to work exactly as before.

### Language behavior

Sarvam STT returns a detected language code. The app uses that language for TTS when supported, with `hi-IN` as the default. This works well for Hindi and Indian-English/code-mixed conversations.

For longer-term production voice agents, Sarvam also provides realtime STT/TTS transports; this implementation deliberately uses the simpler Streamlit microphone + REST flow.
