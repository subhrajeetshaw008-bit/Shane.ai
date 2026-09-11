"""
Jarvis - Steps 1-8: Brain + Voice + Wake Word + Memory + Typed Input + To-Do + Reminders
--------------------------------------------------------------------------------------------
A local, free, JARVIS-style assistant. This version can:
  - Think using a local LLM (Ollama)
  - Speak replies out loud (pyttsx3, offline)
  - Listen to your voice and transcribe it (faster-whisper "base", offline)
  - Wake up automatically when you say "Hey Jarvis" (openwakeword, offline)
  - Remember facts about you permanently across sessions (memory.json)
  - Accept typed messages at any time, as an alternative to voice (Windows only)
  - Manage a to-do list (tasks.json)
  - Set time-based reminders that trigger automatically (reminders.json)

Requirements:
    1. Ollama installed + a model pulled:
           ollama pull phi3
    2. Python packages:
           pip install ollama pyttsx3 faster-whisper sounddevice numpy scipy openwakeword

Usage: either say "Hey Jarvis" out loud and wait for "Yes?", then speak
your request - OR just type your message directly and press Enter.

To save a permanent memory, say or type something like:
    "Remember that I have class at 9 AM"

To-do list commands:
    "Add task buy groceries"       -> adds a new task
    "What are my tasks"            -> lists all tasks with numbers
    "Complete task 2"              -> marks task #2 as done
    "Delete task 2"                -> removes task #2

Reminder commands:
    "Remind me to water the plants at 6 pm"
    "Remind me to call mom at 9:30 pm"
Jarvis checks the clock every ~4 seconds in the background and will
speak the reminder out loud automatically when it's due - you don't
need to ask for it.
"""

import ollama
import pyttsx3
import numpy as np
import sounddevice as sd
import json
import os
import re
import msvcrt  # Windows-only: lets us check for typed input without blocking
from datetime import datetime, timedelta
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


# ---- To-Do List setup ----
TASKS_FILE = "tasks.json"


def load_tasks() -> list[dict]:
    """Load saved tasks from disk. Each task is {'text': ..., 'done': bool}."""
    if not os.path.exists(TASKS_FILE):
        return []
    with open(TASKS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_tasks(tasks: list[dict]):
    with open(TASKS_FILE, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2)


def add_task(text: str):
    tasks = load_tasks()
    tasks.append({"text": text, "done": False})
    save_tasks(tasks)


def list_tasks_text() -> str:
    """Return a spoken-friendly summary of current tasks."""
    tasks = load_tasks()
    if not tasks:
        return "You have no tasks right now."

    lines = []
    for i, task in enumerate(tasks, start=1):
        status = "done" if task["done"] else "not done"
        lines.append(f"{i}. {task['text']} ({status})")
    return "Here are your tasks: " + "; ".join(lines)


def complete_task(index_1_based: int) -> bool:
    """Mark a task as done by its 1-based position. Returns True if successful."""
    tasks = load_tasks()
    if 1 <= index_1_based <= len(tasks):
        tasks[index_1_based - 1]["done"] = True
        save_tasks(tasks)
        return True
    return False


def delete_task(index_1_based: int) -> bool:
    """Delete a task by its 1-based position. Returns True if successful."""
    tasks = load_tasks()
    if 1 <= index_1_based <= len(tasks):
        tasks.pop(index_1_based - 1)
        save_tasks(tasks)
        return True
    return False


# ---- Reminders setup ----
REMINDERS_FILE = "reminders.json"


def load_reminders() -> list[dict]:
    """Each reminder is {'text': ..., 'remind_at': ISO datetime string, 'notified': bool}."""
    if not os.path.exists(REMINDERS_FILE):
        return []
    with open(REMINDERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_reminders(reminders: list[dict]):
    with open(REMINDERS_FILE, "w", encoding="utf-8") as f:
        json.dump(reminders, f, indent=2)


def parse_time_string(time_str: str) -> datetime | None:
    """Parse things like '6 pm', '6:30 pm', '18:00' into a datetime today
    (or tomorrow, if that time has already passed today)."""
    time_str = time_str.strip().lower().replace(".", "")
    formats_to_try = ["%I:%M %p", "%I %p", "%H:%M", "%I:%M%p", "%I%p"]

    for fmt in formats_to_try:
        try:
            parsed_time = datetime.strptime(time_str, fmt)
            now = datetime.now()
            candidate = now.replace(
                hour=parsed_time.hour,
                minute=parsed_time.minute,
                second=0,
                microsecond=0,
            )
            if candidate <= now:
                candidate += timedelta(days=1)  # schedule for tomorrow instead
            return candidate
        except ValueError:
            continue
    return None


def add_reminder(text: str, time_str: str) -> datetime | None:
    """Add a reminder. Returns the scheduled datetime, or None if the time
    couldn't be understood."""
    remind_at = parse_time_string(time_str)
    if remind_at is None:
        return None

    reminders = load_reminders()
    reminders.append({
        "text": text,
        "remind_at": remind_at.isoformat(),
        "notified": False,
    })
    save_reminders(reminders)
    return remind_at


def check_due_reminders() -> list[str]:
    """Return the text of any reminders that are now due, and mark them notified."""
    reminders = load_reminders()
    due_texts = []
    now = datetime.now()
    changed = False

    for reminder in reminders:
        if not reminder["notified"]:
            remind_at = datetime.fromisoformat(reminder["remind_at"])
            if now >= remind_at:
                due_texts.append(reminder["text"])
                reminder["notified"] = True
                changed = True

    if changed:
        save_reminders(reminders)
    return due_texts


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
# since our assistant is named Jarvis). inference_framework="onnx" avoids
# a slower fallback path and the "tflite runtime not found" warning.
oww_model = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
WAKE_WORD_THRESHOLD = 0.3   # lower = easier to trigger (was 0.5)
WAKE_CHUNK_SAMPLES = 1280   # openwakeword expects 80ms chunks at 16kHz
DEBUG_WAKE_SCORES = True    # prints confidence scores so we can tune this


_typed_buffer = ""


def _poll_typed_input():
    """Non-blocking check for typed input. Returns a full line if Enter was
    just pressed, otherwise None. Call this repeatedly in a loop."""
    global _typed_buffer
    if msvcrt.kbhit():
        ch = msvcrt.getwche()  # reads one character and echoes it to screen
        if ch in ("\r", "\n"):
            line = _typed_buffer
            _typed_buffer = ""
            print()  # move to next line after Enter
            return line
        elif ch == "\b":  # backspace
            _typed_buffer = _typed_buffer[:-1]
        else:
            _typed_buffer += ch
    return None


def wait_for_input():
    """Blocks until EITHER 'Hey Jarvis' is heard OR the user types something
    and presses Enter. Also checks for due reminders in the background.
    Returns (text, via_voice)."""
    print(f'[Say "Hey {ASSISTANT_NAME}", or type a message and press Enter...]')
    loop_count = 0
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
                if DEBUG_WAKE_SCORES and score > 0.05:
                    print(f"  (wake score: {score:.2f})")
                if score > WAKE_WORD_THRESHOLD:
                    return None, True  # caller will use listen() for the command

            typed_line = _poll_typed_input()
            if typed_line is not None:
                return typed_line, False

            # Check for due reminders roughly every 4 seconds (50 chunks * 80ms)
            loop_count += 1
            if loop_count % 50 == 0:
                due = check_due_reminders()
                for reminder_text in due:
                    print(f"\n[Reminder!] {reminder_text}")
                    speak(f"Reminder: {reminder_text}")


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
    print(f'Say "Hey {ASSISTANT_NAME}" OR type a message and press Enter.')
    print("Press Ctrl+C to exit.\n")

    # Conversation history - this is what gives it "memory" during the session.
    # build_system_prompt() also injects long-term facts saved across sessions.
    messages = [{"role": "system", "content": build_system_prompt()}]

    while True:
        typed_text, via_voice = wait_for_input()

        if via_voice:
            print("[Wake word detected!]")
            speak("Yes?")
            user_input = listen()
            print(f"You (spoken): {user_input}")
        else:
            user_input = typed_text.strip()
            print(f"You (typed): {user_input}")

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

        # ---- Reminder commands ----
        # e.g. "remind me to water the plants at 6 pm"
        reminder_match = re.match(
            r"remind me to (.+) at (.+)", lowered
        )
        if reminder_match:
            task_text = reminder_match.group(1).strip()
            time_text = reminder_match.group(2).strip()
            scheduled = add_reminder(task_text, time_text)
            if scheduled:
                confirmation = f"Okay, I'll remind you to {task_text} at {scheduled.strftime('%I:%M %p')}."
            else:
                confirmation = f"I couldn't understand the time '{time_text}'. Try something like '6 pm' or '6:30 pm'."
            print(f"{ASSISTANT_NAME}: {confirmation}\n")
            speak(confirmation)
            continue

        # ---- To-Do List commands ----
        if lowered.startswith("add task ") or lowered.startswith("add a task "):
            task_text = user_input.split("task ", 1)[1].strip()
            add_task(task_text)
            confirmation = f"Added to your list: {task_text}."
            print(f"{ASSISTANT_NAME}: {confirmation}\n")
            speak(confirmation)
            continue

        if "my tasks" in lowered or "my to-do" in lowered or "my todo" in lowered:
            summary = list_tasks_text()
            print(f"{ASSISTANT_NAME}: {summary}\n")
            speak(summary)
            continue

        if lowered.startswith("complete task ") or lowered.startswith("finish task "):
            number_part = user_input.split(" ")[-1]
            if number_part.isdigit() and complete_task(int(number_part)):
                confirmation = f"Marked task {number_part} as done."
            else:
                confirmation = f"I couldn't find task {number_part}."
            print(f"{ASSISTANT_NAME}: {confirmation}\n")
            speak(confirmation)
            continue

        if lowered.startswith("delete task ") or lowered.startswith("remove task "):
            number_part = user_input.split(" ")[-1]
            if number_part.isdigit() and delete_task(int(number_part)):
                confirmation = f"Deleted task {number_part}."
            else:
                confirmation = f"I couldn't find task {number_part}."
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
    