#!/usr/bin/env python3
"""
Arjun — Streamlit UI

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import html
import os
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st


# ===========================================================================
# Environment
# ===========================================================================

ROOT = Path(__file__).resolve().parent

# Load .env BEFORE reading environment variables.
load_dotenv(
    ROOT / ".env",
    override=True,
)


# ===========================================================================
# Sarvam configuration
# ===========================================================================

SARVAM_API_KEY = os.environ.get(
    "SARVAM_API_KEY",
    "",
)

SARVAM_STT_MODEL = os.environ.get(
    "SARVAM_STT_MODEL",
    "saaras:v3",
)

SARVAM_STT_MODE = os.environ.get(
    "SARVAM_STT_MODE",
    "transcribe",
)

SARVAM_TTS_MODEL = os.environ.get(
    "SARVAM_TTS_MODEL",
    "bulbul:v3",
)

SARVAM_TTS_SPEAKER = os.environ.get(
    "SARVAM_TTS_SPEAKER",
    "shubh",
)

try:
    SARVAM_TTS_PACE = float(
        os.environ.get(
            "SARVAM_TTS_PACE",
            "1.0",
        )
    )
except ValueError:
    SARVAM_TTS_PACE = 1.0

SARVAM_TTS_LANGUAGE = os.environ.get(
    "SARVAM_TTS_LANGUAGE",
    "hi-IN",
)


# ===========================================================================
# Gemini configuration
# ===========================================================================

api_key = (
    os.environ.get("GEMINI_API_KEY", "")
    or os.environ.get("GOOGLE_API_KEY", "")
)

model = os.environ.get(
    "GEMINI_MODEL",
    "",
)

os.environ["GEMINI_API_KEY"] = api_key
os.environ["GEMINI_MODEL"] = model


# ===========================================================================
# Application configuration
# ===========================================================================

# Charts are generated automatically for each answer.
# This creates one additional LLM call.
SHOW_CHARTS = True


# ===========================================================================
# Paths
# ===========================================================================

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )

FAVICON = (
    ROOT
    / "assets"
    / "my_booth_agent_favicon.png"
)

LOGO = (
    ROOT
    / "assets"
    / "my_booth_agent_logo.png"
)


# ===========================================================================
# Streamlit configuration
# ===========================================================================

st.set_page_config(
    page_title="Arjun",
    page_icon=(
        str(FAVICON)
        if FAVICON.exists()
        else ""
    ),
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ===========================================================================
# Local application imports
# ===========================================================================

import theme  # noqa: E402
from auth import logout_button, require_login  # noqa: E402
from gyanpur_booth.textutil import to_plain_text  # noqa: E402


theme.apply()


# ===========================================================================
# Login gate
# ===========================================================================

if not require_login():
    st.stop()


# ===========================================================================
# Session state
# ===========================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "voice_language_code" not in st.session_state:
    st.session_state.voice_language_code = SARVAM_TTS_LANGUAGE


# ===========================================================================
# Agent helpers
# ===========================================================================

@st.cache_resource(show_spinner=False)
def get_agent(
    api_key_value: str,
    model_name: str,
):
    from gyanpur_booth.common import GeminiClient
    from gyanpur_booth.booth_agent import BoothAgent

    client = GeminiClient(
        api_key=api_key_value,
        model=model_name,
    )

    return BoothAgent(
        llm=client,
    )


@st.cache_resource(show_spinner=False)
def get_viz_agent(
    api_key_value: str,
    model_name: str,
):
    from gyanpur_booth.common import GeminiClient
    from gyanpur_booth.visualization_agent import VisualizationAgent

    client = GeminiClient(
        api_key=api_key_value,
        model=model_name,
    )

    return VisualizationAgent(
        llm=client,
    )


# ===========================================================================
# Voice helpers
# ===========================================================================

def voice_available() -> bool:
    """Return True when Sarvam API key is configured."""

    return bool(
        SARVAM_API_KEY.strip()
    )


def synthesize_answer_audio(
    text: str,
    language_code: str | None = None,
) -> bytes:
    """
    Generate TTS audio.

    Voice failure must never break the main text response.
    """

    from sarvam_voice import text_to_speech

    lang = (
        language_code
        or SARVAM_TTS_LANGUAGE
    )

    supported = {
        "hi-IN",
        "en-IN",
        "bn-IN",
        "ta-IN",
        "te-IN",
        "kn-IN",
        "ml-IN",
        "mr-IN",
        "gu-IN",
        "pa-IN",
        "od-IN",
    }

    if lang not in supported:
        lang = SARVAM_TTS_LANGUAGE

    return text_to_speech(
        text,
        SARVAM_API_KEY,
        language_code=lang,
        model=SARVAM_TTS_MODEL,
        speaker=SARVAM_TTS_SPEAKER,
        pace=SARVAM_TTS_PACE,
    )


def render_voice_output(
    audio_bytes: bytes,
    key: str,
) -> None:
    """Render Sarvam WAV response in browser."""

    if audio_bytes:
        st.audio(
            audio_bytes,
            format="audio/wav",
            autoplay=True,
        )


# ===========================================================================
# Text rendering helpers
# ===========================================================================

def render_plain(text: str) -> None:
    """
    Show text as plain text.

    No Markdown.
    No LaTeX.
    No emphasis.
    """

    safe = (
        html.escape(
            to_plain_text(text)
        )
        .replace(
            "$",
            "&#36;",
        )
        .replace(
            "\n",
            "<br>",
        )
    )

    st.markdown(
        f'<div class="plain">{safe}</div>',
        unsafe_allow_html=True,
    )


def role_label(role: str) -> None:
    """Render chat role label."""

    label = (
        "You"
        if role == "user"
        else "Arjun"
    )

    st.markdown(
        f'<div class="role">{label}</div>',
        unsafe_allow_html=True,
    )


# ===========================================================================
# Chart rendering
# ===========================================================================

def render_charts(
    charts,
    key_prefix: str,
) -> None:
    """
    Draw chart specifications from Visualization Agent.

    Plotly is preferred.
    Matplotlib is used as fallback.
    """

    from gyanpur_booth import chart_renderer

    for i, chart in enumerate(
        charts or []
    ):
        key = f"{key_prefix}_{i}"

        try:
            try:
                fig = chart_renderer.to_plotly(
                    chart
                )

                try:
                    st.plotly_chart(
                        fig,
                        width="stretch",
                        theme=None,
                        key=key,
                    )
                except TypeError:
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        theme=None,
                        key=key,
                    )

            except ImportError:
                fig = chart_renderer.to_matplotlib(
                    chart
                )

                try:
                    st.pyplot(
                        fig,
                        width="stretch",
                    )
                except TypeError:
                    st.pyplot(
                        fig,
                        use_container_width=True,
                    )

        except Exception as e:
            print(
                "[ui] could not draw chart "
                f"{chart.get('title', '')!r}: {e}"
            )


# ===========================================================================
# Header
# ===========================================================================

st.markdown(
    """
<div class="brand">
  <div style="display:flex;align-items:center;gap:.8rem;">
    <p class="name">Arjun</p>
  </div>
  <p class="sub">
    Electoral intelligence and data analysis for Gyanpur Vidhan Sabha
  </p>
  <p class="meta">
    Voter roll · Election history · Booth portfolios · Data-backed analysis
  </p>
</div>
""",
    unsafe_allow_html=True,
)


# ===========================================================================
# Suggested questions
# ===========================================================================

EXAMPLES = [
    "ज्ञानपुर में भाजपा कितनी बार जीती है?",
    "2022 में खोखर में कौन जीता था?",
    "खोखर में यादव मतदाताओं की संख्या कितनी है?",
    "पार्ट नंबर 1 का जनसांख्यिकीय विश्लेषण दिखाइए।",
    "ज्ञानपुर के किन बूथों में सबसे अधिक मतदाता हैं?",
    "आप मेरे लिए किस तरह का विश्लेषण कर सकते हैं?",
]


st.markdown(
    '<div class="section-label">उदाहरण प्रश्न</div>',
    unsafe_allow_html=True,
)

cols = st.columns(2)

for i, example in enumerate(EXAMPLES):
    with cols[i % 2]:
        if st.button(
            example,
            key=f"ex_{i}",
            use_container_width=True,
        ):
            st.session_state.pending_prompt = example


st.write("")


# ===========================================================================
# Voice input
# ===========================================================================

st.markdown(
    '<div class="section-label">Talk to Arjun</div>',
    unsafe_allow_html=True,
)


if voice_available():

    st.markdown(
        """
<div class="voice-card">
  <div class="voice-title">🎙️ Ask Arjun by voice</div>
  <div class="voice-help">
    Press the microphone button to start recording.
    Stop the recording when you finish speaking,
    and Arjun will analyse your question and reply.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    audio_prompt = st.audio_input(
        "Record question",
        key="arjun_microphone",
    )

    if audio_prompt is not None:

        # ---------------------------------------------------------------
        # Process this recording only once.
        # ---------------------------------------------------------------

        audio_id = (
            getattr(
                audio_prompt,
                "file_id",
                None,
            )
            or str(
                (
                    getattr(
                        audio_prompt,
                        "name",
                        None,
                    ),
                    getattr(
                        audio_prompt,
                        "size",
                        None,
                    ),
                )
            )
        )

        if (
            st.session_state.get(
                "last_audio_id"
            )
            != audio_id
        ):

            st.session_state.last_audio_id = audio_id

            try:

                from sarvam_voice import speech_to_text

                # -------------------------------------------------------
                # STT
                #
                # speech_to_text() now reads the Streamlit bytes and
                # constructs its own multipart request:
                #
                #     filename = audio.wav
                #     MIME     = audio/wav
                #
                # This avoids the audio/vnd.wave problem.
                # -------------------------------------------------------

                with st.spinner(
                    "Arjun is transcribing your question..."
                ):

                    transcript, detected_language = (
                        speech_to_text(
                            audio_prompt,
                            SARVAM_API_KEY,
                            model=SARVAM_STT_MODEL,
                            mode=SARVAM_STT_MODE,
                        )
                    )

                if transcript:

                    st.session_state.voice_language_code = (
                        detected_language
                    )

                    st.session_state.pending_prompt = (
                        transcript
                    )

                    st.rerun()

                else:

                    st.warning(
                        "I could not detect any speech. "
                        "Please try again."
                    )

            except Exception as e:

                traceback.print_exc()

                st.error(
                    f"Voice input failed: {e}"
                )

else:

    st.caption(
        "Voice is disabled. Add SARVAM_API_KEY "
        "to Streamlit secrets to enable Arjun voice mode."
    )


st.write("")


# ===========================================================================
# Conversation history
# ===========================================================================

for idx, msg in enumerate(
    st.session_state.messages
):

    with st.chat_message(
        msg["role"]
    ):

        role_label(
            msg["role"]
        )

        render_plain(
            msg["content"]
        )

        if msg.get("charts"):
            render_charts(
                msg["charts"],
                key_prefix=f"chart_{idx}",
            )


# ===========================================================================
# Chat input
# ===========================================================================

prompt = st.chat_input(
    "Ask Arjun about voters, results, or booth intelligence…"
)


# Voice input can inject a prompt into the normal chat pipeline.
if "pending_prompt" in st.session_state:

    prompt = st.session_state.pop(
        "pending_prompt"
    )


# ===========================================================================
# Process user prompt
# ===========================================================================

if prompt:

    # ---------------------------------------------------------------
    # Store user message
    # ---------------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):

        role_label("user")

        render_plain(
            prompt
        )


    # ---------------------------------------------------------------
    # Arjun response
    # ---------------------------------------------------------------

    with st.chat_message("assistant"):

        role_label("assistant")

        try:

            # -----------------------------------------------------------
            # Main agent
            # -----------------------------------------------------------

            with st.spinner(
                "Analysing the data..."
            ):

                answer = to_plain_text(
                    get_agent(
                        api_key,
                        model,
                    ).run(
                        prompt
                    )
                )

            render_plain(
                answer
            )


            # -----------------------------------------------------------
            # Sarvam TTS
            #
            # Voice generation is deliberately isolated from the main
            # answer. If TTS fails, the written answer still works.
            # -----------------------------------------------------------

            voice_audio = b""

            if voice_available():

                try:

                    with st.spinner(
                        "Arjun is preparing a short consultant brief..."
                    ):

                        voice_brief, voice_language = (
                            get_agent(
                                api_key,
                                model,
                            ).voice_brief(
                                prompt,
                                answer,
                                language_hint=(
                                    st.session_state.get(
                                        "voice_language_code"
                                    )
                                ),
                            )
                        )

                        voice_audio = (
                            synthesize_answer_audio(
                                voice_brief,
                                voice_language
                                or st.session_state.get(
                                    "voice_language_code",
                                    SARVAM_TTS_LANGUAGE,
                                ),
                            )
                        )

                    if voice_audio:

                        st.caption(
                            "Arjun's spoken analysis"
                        )

                        render_voice_output(
                            voice_audio,
                            key=(
                                f"voice_"
                                f"{len(st.session_state.messages)}"
                            ),
                        )

                except Exception:

                    # TTS failure must never break text response.
                    traceback.print_exc()


            # -----------------------------------------------------------
            # Visualization Agent
            # -----------------------------------------------------------

            charts = []

            if SHOW_CHARTS:

                try:

                    with st.spinner(
                        "Preparing charts..."
                    ):

                        viz = (
                            get_viz_agent(
                                api_key,
                                model,
                            ).analyse(
                                answer,
                                prompt,
                            )
                        )

                    charts = (
                        viz.get(
                            "charts",
                            [],
                        )
                        if viz.get(
                            "has_charts"
                        )
                        else []
                    )

                except Exception:

                    # Chart failure must never break text response.
                    traceback.print_exc()


            # -----------------------------------------------------------
            # Render charts
            # -----------------------------------------------------------

            if charts:

                render_charts(
                    charts,
                    key_prefix=(
                        f"chart_"
                        f"{len(st.session_state.messages)}"
                    ),
                )


            # -----------------------------------------------------------
            # Save assistant message
            # -----------------------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "charts": charts,
                }
            )


        except Exception:

            traceback.print_exc()

            err = (
                "Something went wrong while processing "
                "that question. Please try again."
            )

            render_plain(
                err
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": err,
                }
            )


# ===========================================================================
# Bottom controls
# ===========================================================================

st.divider()

control_cols = st.columns(
    [1, 1, 4]
)


with control_cols[0]:

    if st.button(
        "Clear chat",
        use_container_width=True,
        key="clear_chat_bottom",
    ):

        for key in (
            "messages",
            "pending_prompt",
            "last_audio_id",
            "voice_language_code",
        ):

            st.session_state.pop(
                key,
                None,
            )

        st.rerun()


with control_cols[1]:

    logout_button()


st.caption(
    "Arjun · myboothagent.com"
)