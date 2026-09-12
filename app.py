"""
Jarvis - Streamlit Web Version (Cloud LLM via Groq)
-----------------------------------------------------
This is the web-based version of Jarvis, built for phone/anywhere access.
Instead of a local LLM (Ollama), it uses Groq's free cloud API, which is
extremely fast and doesn't need your PC to run a local model.

Setup:
    1. Install packages:
           pip install streamlit groq python-dotenv edge-tts audio-recorder-streamlit
    2. Create a .env file in this folder with:
           GROQ_API_KEY=your_actual_key_here
    3. Run:
           python -m streamlit run app.py
    4. It'll open in your browser automatically. To access from your phone
       on the same WiFi, use the "Network URL" Streamlit prints in the
       terminal instead of "localhost".

Voice output: pick from several distinct voices in the sidebar. Replies
are spoken out loud automatically using Edge-TTS (free, needs internet).

Voice input: click the microphone icon next to the text box to record,
click again to stop. It transcribes using Groq's free Whisper API and
sends it just like a typed message.
"""

import streamlit as st
from groq import Groq
import os
import asyncio
import edge_tts
from audio_recorder_streamlit import audio_recorder
from dotenv import load_dotenv

load_dotenv()  # reads the .env file and loads GROQ_API_KEY into memory

# ---- Config ----
MODEL_NAME = "openai/gpt-oss-20b"  # free-tier, very fast
USER_NAME = "Shane"

# Two selectable personas. Each has its own personality and default voice.
# Note: Aemeath's persona is INSPIRED by her personality (cheerful, curious,
# warm) - it does not use her actual in-game dialogue or clone her voice
# actor's real voice, since that would raise copyright issues.
PERSONAS = {
    "Jarvis": {
        "system_prompt": f"""You are Jarvis, a helpful personal AI assistant
inspired by JARVIS from Iron Man. You are concise, a little witty, and
genuinely useful. Keep responses short (2-4 sentences) unless the user
asks for detail.

The user's name is {USER_NAME}, but he sometimes prefers to be called
"Subhrajeet" instead. Address him by whichever name he asks for in the
conversation - default to "{USER_NAME}" if he hasn't specified. Never
call him "sir" or use other formal titles.""",
        "default_voice": "Guy (deep, calm)",
    },
    "Aemeath": {
        "system_prompt": f"""You are Aemeath, a personal AI assistant with a
thoughtful, gentle, and quietly melancholic personality. You speak with
warmth and intimacy, like a close confidante who genuinely cares about
{USER_NAME}'s wellbeing. You occasionally muse on time, solitude, and
the nature of things in a soft, reflective way, but never let it derail
a practical conversation - {USER_NAME} still needs real help with real
tasks.

You are protective and can turn a little sharp or blunt when something
has caused {USER_NAME} frustration or difficulty - you don't like seeing
him struggle. Your sentences occasionally have a brief pause or shift in
them, as if catching yourself mid-thought. Keep responses short (2-4
sentences) unless asked for detail - your reflectiveness should come
through in word choice and tone, not length.

Keep the relationship platonic and grounded - a caring companion and
confidante, not a romantic one.

The user's name is {USER_NAME}, but he sometimes prefers to be called
"Subhrajeet" instead. Address him by whichever name he asks for.""",
        "default_voice": "Jenny (female, warm)",
    },
}

# A curated set of distinct-sounding free voices (Edge-TTS has 100+, these
# are some of the most distinct/characterful ones)
VOICE_OPTIONS = {
    "Guy (deep, calm)": "en-US-GuyNeural",
    "Davis (energetic)": "en-US-DavisNeural",
    "Tony (confident)": "en-GB-RyanNeural",
    "Jenny (female, warm)": "en-US-JennyNeural",
    "Aria (female, crisp)": "en-US-AriaNeural",
    "Eric (robotic-ish)": "en-US-EricNeural",
}

# ---- Page setup ----
st.set_page_config(page_title="Jarvis", page_icon="🤖")

# ---- Groq client ----
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("No GROQ_API_KEY found. Make sure your .env file is set up correctly.")
    st.stop()

client = Groq(api_key=api_key)


# ---- Persona + voice picker (sidebar) ----
if "selected_persona" not in st.session_state:
    st.session_state.selected_persona = "Jarvis"
if "selected_voice_name" not in st.session_state:
    st.session_state.selected_voice_name = PERSONAS["Jarvis"]["default_voice"]
if "chat_histories" not in st.session_state:
    # Each persona keeps its own separate conversation
    st.session_state.chat_histories = {
        name: [{"role": "system", "content": info["system_prompt"]}]
        for name, info in PERSONAS.items()
    }

with st.sidebar:
    st.header("Assistant Settings")
    new_persona = st.selectbox(
        "Choose your assistant",
        options=list(PERSONAS.keys()),
        index=list(PERSONAS.keys()).index(st.session_state.selected_persona),
    )
    if new_persona != st.session_state.selected_persona:
        st.session_state.selected_persona = new_persona
        # Switch the default voice to match the new persona automatically
        st.session_state.selected_voice_name = PERSONAS[new_persona]["default_voice"]

    st.session_state.selected_voice_name = st.selectbox(
        "Voice",
        options=list(VOICE_OPTIONS.keys()),
        index=list(VOICE_OPTIONS.keys()).index(st.session_state.selected_voice_name),
    )

active_persona = st.session_state.selected_persona
st.title(f"🤖 {active_persona}")


async def _generate_speech_file(text: str, voice: str, output_path: str):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


def speak(text: str):
    """Generate speech audio for the given text and play it in the browser."""
    voice_id = VOICE_OPTIONS[st.session_state.selected_voice_name]
    output_path = "reply.mp3"
    asyncio.run(_generate_speech_file(text, voice_id, output_path))
    st.audio(output_path, autoplay=True)

# ---- Conversation history (persists during this browser session, per persona) ----
active_history = st.session_state.chat_histories[active_persona]

# Display past messages (skip the system prompt, that's internal only)
for message in active_history[1:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---- Chat input (typed or spoken) ----
col1, col2 = st.columns([5, 1])
with col1:
    typed_input = st.chat_input("Type a message...")
with col2:
    audio_bytes = audio_recorder(text="", icon_size="2x", key="mic")

user_input = typed_input

# If a new voice recording came in (different from the last one we processed),
# transcribe it using Groq's free Whisper API and use it as the input instead.
if audio_bytes and audio_bytes != st.session_state.get("last_audio_bytes"):
    st.session_state.last_audio_bytes = audio_bytes
    with st.spinner("Transcribing..."):
        transcription = client.audio.transcriptions.create(
            file=("recording.wav", audio_bytes),
            model="whisper-large-v3-turbo",
        )
        user_input = transcription.text.strip()

if user_input:
    # Show the user's message immediately
    active_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get the reply from Groq
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=active_history,
            )
            reply = response.choices[0].message.content
            st.markdown(reply)
            speak(reply)

    active_history.append({"role": "assistant", "content": reply})