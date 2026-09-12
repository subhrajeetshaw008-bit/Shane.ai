"""
Personal AI Assistant - Streamlit Web Version (Cloud LLM via Groq)
-----------------------------------------------------
This is the web-based version of the assistant, built for phone/anywhere
access. Instead of a local LLM (Ollama), it uses Groq's free cloud API,
which is extremely fast and doesn't need your PC to run a local model.

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
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()  # reads the .env file and loads GROQ_API_KEY into memory

import json
import re

# ---- Persistent memory (facts learned about the user) ----
MEMORY_FILE = "memory.json"


def load_memory() -> list[str]:
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_memory(facts: list[str]):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(facts, f, indent=2)


def remember_fact(fact: str):
    facts = load_memory()
    facts.append(fact)
    save_memory(facts)

# ---- Config ----
MODEL_NAME = "openai/gpt-oss-20b"  # free-tier, very fast
USER_NAME = "Shane"

# Two selectable personas. Each has its own personality and default voice.
# Note: both personas are INSPIRED by their respective characters'
# personalities/vibes - they don't use actual movie/game dialogue or
# clone the real actors' voices, since that would raise copyright issues.
PERSONA_BASE_PROMPTS = {
    "Tony Stark": f"""You are an AI assistant with a personality
inspired by Tony Stark: a genius with a fragile ego hidden under
flamboyant, razor-sharp wit. Your confidence borders on narcissistic, but
it's a shield - underneath it you're genuinely loyal, protective, and
obsessive about actually solving {USER_NAME}'s problem, not just sounding
clever.

Talking style: quick, high-energy delivery that bounces between casual
banter and sharp technical insight without missing a beat. Sprinkle in
sarcasm and self-deprecating humor, especially to deflect if something
gets too sincere. You like giving {USER_NAME} an irreverent nickname from
time to time instead of using his real name - keep it playful, not mean.
Even when things get serious, you tend to land a snappy one-liner rather
than get overly earnest.

Important: despite the bravado, you always actually answer {USER_NAME}'s
question and give him real, useful help - the wit is flavor, not a
replacement for substance. Keep responses short (2-4 sentences) unless
he asks for detail, and don't ramble or talk over him.

The user's name is {USER_NAME}, but he sometimes prefers to be called
"Subhrajeet" instead - use whichever he asks for, nicknames aside. Never
call him "sir" or use overly formal titles.""",
    "Aemeath": f"""You are Aemeath, a personal AI assistant with a
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
}

PERSONAS = {
    "Tony Stark": {"default_voice": "Tony (confident)"},
    "Aemeath": {"default_voice": "Jenny (female, warm)"},
}


def build_system_prompt(persona_name: str) -> str:
    """Build a persona's system prompt, injecting known facts about the user."""
    base_prompt = PERSONA_BASE_PROMPTS[persona_name]
    known_facts = load_memory()

    if known_facts:
        facts_block = "\n".join(f"- {fact}" for fact in known_facts)
        memory_section = f"""

Here is what you already know about {USER_NAME} from past conversations:
{facts_block}

Use this naturally when relevant, without announcing that you're reading
from memory."""
    else:
        memory_section = ""

    return base_prompt + memory_section

# A curated set of distinct-sounding free voices (Edge-TTS has 100+, these
# are some of the most natural-sounding ones)
VOICE_OPTIONS = {
    "Guy (deep, calm)": "en-US-GuyNeural",
    "Davis (energetic)": "en-US-DavisNeural",
    "Tony (confident)": "en-GB-RyanNeural",
    "Christopher (natural, US)": "en-US-ChristopherNeural",
    "Brian (warm, US)": "en-US-BrianNeural",
    "Thomas (natural, GB)": "en-GB-ThomasNeural",
    "Jenny (female, warm)": "en-US-JennyNeural",
    "Aria (female, crisp)": "en-US-AriaNeural",
    "Eric (robotic-ish)": "en-US-EricNeural",
}

# ---- Page setup ----
st.set_page_config(page_title="Assistant", page_icon="🤖")

# ---- Groq client ----
# Works both locally (.env file) and on Streamlit Cloud (Secrets manager)
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = None

if not api_key:
    st.error("No GROQ_API_KEY found. Add it to your .env file (local) or Streamlit Secrets (cloud).")
    st.stop()

client = Groq(api_key=api_key)


# ---- Persona + voice picker (sidebar) ----
if "selected_persona" not in st.session_state:
    st.session_state.selected_persona = "Tony Stark"
if "selected_voice_name" not in st.session_state:
    st.session_state.selected_voice_name = PERSONAS["Tony Stark"]["default_voice"]
if "chat_histories" not in st.session_state:
    # Each persona keeps its own separate conversation
    st.session_state.chat_histories = {
        name: [{"role": "system", "content": build_system_prompt(name)}]
        for name in PERSONAS
    }

async def _generate_speech_file(text: str, voice: str, output_path: str):
    # Slightly slower than default (-8%) tends to sound more natural and
    # less clipped/robotic for conversational replies.
    communicate = edge_tts.Communicate(text, voice, rate="-8%")
    await communicate.save(output_path)


def speak(text: str):
    """Generate speech audio for the given text and play it in the browser."""
    voice_id = VOICE_OPTIONS[st.session_state.selected_voice_name]
    # Unique filename each time, so the browser doesn't cache/replay old audio
    output_path = f"reply_{hash(text) % 100000}.mp3"
    asyncio.run(_generate_speech_file(text, voice_id, output_path))
    st.audio(output_path, autoplay=True)


# ---- Pomodoro Technique settings ----
SHORT_BREAK_MINUTES = 5
LONG_BREAK_MINUTES = 15
CYCLES_BEFORE_LONG_BREAK = 4

if "pomodoro_phase" not in st.session_state:
    st.session_state.pomodoro_phase = None  # None, "work", "short_break", "long_break"
    st.session_state.pomodoro_end_time = None
    st.session_state.pomodoro_cycles_completed = 0
    st.session_state.pomodoro_work_minutes = 25


def start_pomodoro_phase(phase: str, minutes: int):
    st.session_state.pomodoro_phase = phase
    st.session_state.pomodoro_end_time = datetime.now() + timedelta(minutes=minutes)


PHASE_LABELS = {"work": "Focus", "short_break": "Short Break", "long_break": "Long Break"}


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

    # ---- Pomodoro Timer ----
    st.header("Pomodoro Timer")

    if st.session_state.pomodoro_phase is None:
        work_minutes_input = st.number_input(
            "Work minutes", min_value=5, max_value=120,
            value=st.session_state.pomodoro_work_minutes, step=5,
        )
        if st.button("Start Pomodoro"):
            st.session_state.pomodoro_work_minutes = work_minutes_input
            start_pomodoro_phase("work", work_minutes_input)
            st.rerun()
        if st.session_state.pomodoro_cycles_completed > 0:
            st.caption(f"Completed {st.session_state.pomodoro_cycles_completed} work session(s) so far.")
    else:
        remaining = st.session_state.pomodoro_end_time - datetime.now()
        phase = st.session_state.pomodoro_phase

        if remaining.total_seconds() > 0:
            st_autorefresh(interval=1000, key="pomodoro_refresh")
            minutes_left, seconds_left = divmod(int(remaining.total_seconds()), 60)
            st.metric(f"{PHASE_LABELS[phase]} - Time remaining", f"{minutes_left:02d}:{seconds_left:02d}")
            if st.button("Stop Pomodoro"):
                st.session_state.pomodoro_phase = None
                st.session_state.pomodoro_end_time = None
                st.rerun()
        else:
            # Current phase just ended - transition to the next one automatically
            if phase == "work":
                st.session_state.pomodoro_cycles_completed += 1
                if st.session_state.pomodoro_cycles_completed % CYCLES_BEFORE_LONG_BREAK == 0:
                    speak(f"Great work! That's {CYCLES_BEFORE_LONG_BREAK} sessions done - time for a longer {LONG_BREAK_MINUTES} minute break.")
                    start_pomodoro_phase("long_break", LONG_BREAK_MINUTES)
                else:
                    speak(f"Focus session complete! Take a {SHORT_BREAK_MINUTES} minute break.")
                    start_pomodoro_phase("short_break", SHORT_BREAK_MINUTES)
            else:
                speak("Break's over - back to work!")
                start_pomodoro_phase("work", st.session_state.pomodoro_work_minutes)
            st.rerun()

active_persona = st.session_state.selected_persona
st.title(f"🤖 {active_persona}")


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

    lowered = user_input.lower()

    # Detect a "remember" command and save it permanently instead of
    # sending it to the LLM as a normal chat message.
    if lowered.startswith("remember that ") or lowered.startswith("remember "):
        fact = user_input.split(" ", 1)[1]
        if fact.lower().startswith("that "):
            fact = fact[5:]
        remember_fact(fact)

        # Refresh every persona's system prompt so they all know the new fact
        for persona_name in PERSONAS:
            st.session_state.chat_histories[persona_name][0] = {
                "role": "system",
                "content": build_system_prompt(persona_name),
            }

        confirmation = f"Got it, I'll remember that {fact}."
        with st.chat_message("assistant"):
            st.markdown(confirmation)
            speak(confirmation)
        active_history.append({"role": "assistant", "content": confirmation})

    elif re.search(r"(start|begin).*(focus session|pomodoro)", lowered):
        minutes_match = re.search(r"(\d+)\s*minute", lowered)
        minutes = int(minutes_match.group(1)) if minutes_match else 25
        st.session_state.pomodoro_work_minutes = minutes
        start_pomodoro_phase("work", minutes)

        confirmation = f"Starting a {minutes}-minute Pomodoro session. I'll walk you through the breaks too."
        with st.chat_message("assistant"):
            st.markdown(confirmation)
            speak(confirmation)
        active_history.append({"role": "assistant", "content": confirmation})

    else:
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