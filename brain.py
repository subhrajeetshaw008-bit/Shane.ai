"""
JARVIS-style assistant - Step 1: The Brain (text chat with local LLM)
--------------------------------------------------------------------
This is the core conversation loop. No voice yet - just proving the
"brain" works before we bolt on ears (speech-to-text) and a mouth
(text-to-speech).
 
Requirements:
    1. Install Ollama from https://ollama.com (free, works on Windows)
    2. Pull a small model that fits in 8GB RAM, e.g.:
           ollama pull phi3
       or
           ollama pull llama3.2:3b
    3. Install the python package:
           pip install ollama
    4. Run this file:
           python brain.py
"""
 
import ollama
import pyttsx3
 
# ---- Config ----
MODEL_NAME = "phi3"  # change to "llama3.2:3b" if you pulled that instead
ASSISTANT_NAME = "Shane.ai"
 
 
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
 
SYSTEM_PROMPT = f"""You are {ASSISTANT_NAME}, a helpful personal AI assistant
inspired by JARVIS from Iron Man. You are concise, a little witty, and
genuinely useful. Keep responses short (2-4 sentences) unless the user
asks for detail, since your replies will eventually be spoken out loud."""
 
 
def chat_loop():
    print(f"{ASSISTANT_NAME} is online. Type 'quit' to exit.\n")
 
    # Conversation history - this is what gives it "memory" during the session
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
 
    while True:
        user_input = input("You: ").strip()
 
        if user_input.lower() in ("quit", "exit", "bye"):
            farewell = "Goodbye, sir."
            print(f"{ASSISTANT_NAME}: {farewell}")
            speak(farewell)
            break
 
        if not user_input:
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
 