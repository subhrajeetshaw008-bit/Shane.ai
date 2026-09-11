"""
Jarvis - Steps 1-5: Brain + Voice Output + Voice Input + Wake Word + Memory
----------------------------------------------------------------------------
A local, free, JARVIS-style assistant. This version can:
  - Think using a local LLM (Ollama)
  - Speak replies out loud (pyttsx3, offline)
  - Listen to your voice and transcribe it (faster-whisper "base", offline)
  - Wake up automatically when you say "Hey Jarvis" (openwakeword, offline)
  - Remember facts about you permanently across sessions (memory.json)

Requirements:
    1. Ollama installed + a model pulled:
           ollama pull phi3
    2. Python packages:
           pip install ollama pyttsx3 faster-whisper sounddevice numpy scipy openwakeword
    3. Run:
           python brain.py

Usage: just say "Hey Jarvis" out loud, wait for it to say "Yes?", then
speak your request. This version listens for voice only - typing isn't
available while it's waiting for the wake word.

To save a permanent memory, say something like:
    "Remember that I have class at 9 AM"
    "Remember I don't like spicy food"
It'll confirm out loud, and bring that fact into every future session.
"""

import ollama
import pyttsx3
import numpy as np
import sounddevice as sd
import json
import os
from faster_whisper import WhisperModel
import openwakeword
from openwakeword.model import Model

# Downloads the pre-trained wake word models the first time this runs.
# After that, it's cached locally - no repeated downloads.
openwakeword.utils.download_models()

# ---- Config ----
MODEL_NAME = "phi3"  # change to "llama3.2:3b" if you pulled that instead
ASSISTANT_NAME = "Jarvis"
USER_NAME = "Shane"  # change to "Subhrajeet" if you want that as the default

# ---- Memory setup ----
# Facts Jarvis learns about you are stored here, permanently, between runs.
MEMORY_FILE = "memory.json"


def load_memory() -> list[str]:
    """Load saved facts from disk. Returns an empty list if none exist yet."""
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_memory(facts: list[str]):
    """Write the current list of facts to disk."""
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(facts, f, indent=2)


def remember_fact(fact: str):
    """Add a new fact to memory and save it immediately."""
    facts = load_memory()
    facts.append(fact)
    save_memory(facts)


def speak(text: str):
    """Make Shane.ai say the given text out loud.

    Note: on Windows, reusing a single pyttsx3 engine across multiple
    calls can silently stop working after the first one. Creating a
    fresh engine each time avoids that bug.
    """
    engine = pyttsx3.init()
    engine.setProperty("rate", 175)    # speaking speed (words per minute)
    engine.setProperty("volume", 1.0)  # 0.0 to 1.0
    engine.say(text)
    engine.runAndWait()
    engine.stop()


# ---- Speech-to-Text setup ----
# "base" gives noticeably better accuracy than "tiny" while still
# running fine on 8GB RAM. Try "small" later if you want even better
# accuracy and don't mind a bit more lag.
WHISPER_MODEL_SIZE = "base"
stt_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")

SAMPLE_RATE = 16000  # Whisper expects 16kHz audio
RECORD_SECONDS = 5   # how long to listen after you press Enter


def listen() -> str:
    """Record a few seconds of audio from the mic and transcribe it to text."""
    print(f"[Listening for {RECORD_SECONDS} seconds... speak now]")
    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()  # block until recording finishes

    audio = np.squeeze(audio)
    segments, _ = stt_model.transcribe(
        audio,
        language="en",
        beam_size=5,              # searches more possibilities = more accurate
        vad_filter=True,          # strips out silence/background noise
        condition_on_previous_text=False,  # avoids repeating past mistakes
    )
    text = " ".join(segment.text for segment in segments).strip()
    return text


# ---- Wake Word setup ----
# Uses the pre-trained "hey_jarvis" model (free, offline, no training needed
# since our assistant is named Jarvis).
oww_model = Model(wakeword_models=["hey_jarvis"])
WAKE_WORD_THRESHOLD = 0.5   # confidence needed to trigger (0.0-1.0)
WAKE_CHUNK_SAMPLES = 1280   # openwakeword expects 80ms chunks at 16kHz


def wait_for_wake_word():
    """Block until 'Hey Jarvis' is heard, listening continuously in the background."""
    print(f'[Waiting for you to say "Hey {ASSISTANT_NAME}"...]')
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=WAKE_CHUNK_SAMPLES,
    ) as stream:
        while True:
            chunk, _ = stream.read(WAKE_CHUNK_SAMPLES)
            chunk = chunk.flatten()
            predictions = oww_model.predict(chunk)

            for _, score in predictions.items():
                if score > WAKE_WORD_THRESHOLD:
                    return

def build_system_prompt() -> str:
    """Build the system prompt, injecting whatever Jarvis has learned so far."""
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

    return f"""You are {ASSISTANT_NAME}, a helpful personal AI assistant
inspired by JARVIS from Iron Man. You are concise, a little witty, and
genuinely useful. Keep responses short (2-4 sentences) unless the user
asks for detail, since your replies will eventually be spoken out loud.

The user's name is {USER_NAME}, but he sometimes prefers to be called
"Subhrajeet" instead. Address him by whichever name he asks for in the
conversation - default to "{USER_NAME}" if he hasn't specified. Never
call him "sir" or use other formal titles.{memory_section}"""


def chat_loop():
    print(f"{ASSISTANT_NAME} is online.")
    print(f'Say "Hey {ASSISTANT_NAME}" to talk. Press Ctrl+C to exit.\n')

    # Conversation history - this is what gives it "memory" during the session.
    # build_system_prompt() also injects long-term facts saved across sessions.
    messages = [{"role": "system", "content": build_system_prompt()}]

    while True:
        wait_for_wake_word()
        print("[Wake word detected!]")
        speak("Yes?")
        user_input = listen()
        print(f"You (spoken): {user_input}")

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "bye"):
            farewell = f"Goodbye, {USER_NAME}."
            print(f"{ASSISTANT_NAME}: {farewell}")
            speak(farewell)
            break

        # Detect a "remember" command and save it permanently instead of
        # sending it to the LLM as a normal chat message.
        lowered = user_input.lower()
        if lowered.startswith("remember that ") or lowered.startswith("remember "):
            fact = user_input.split(" ", 1)[1]
            if fact.lower().startswith("that "):
                fact = fact[5:]
            remember_fact(fact)

            # Refresh the system prompt so this session also knows the new fact
            messages[0] = {"role": "system", "content": build_system_prompt()}

            confirmation = f"Got it, I'll remember that {fact}."
            print(f"{ASSISTANT_NAME}: {confirmation}\n")
            speak(confirmation)
            continue

        messages.append({"role": "user", "content": user_input})

        # Send the full conversation so far to the local model
        response = ollama.chat(model=MODEL_NAME, messages=messages)
        reply = response["message"]["content"]

        print(f"{ASSISTANT_NAME}: {reply}\n")
        speak(reply)

        # Keep the assistant's reply in history too, so it remembers context
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    chat_loop()