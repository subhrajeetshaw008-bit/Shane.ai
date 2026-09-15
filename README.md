# Shane.ai

A persistent-memory, voice-driven personal AI assistant, inspired by JARVIS-style companions.

## Features
- 🎙️ Full conversational AI powered by a free cloud LLM (Groq)
- 🔊 Text-to-speech voice output (Edge-TTS) with multiple voice options
- 🧠 Persistent memory — remembers facts about the user across sessions via natural "remember that..." commands
- ✅ To-do list management (add/list/complete/delete tasks)
- ⏰ Time-based reminders, spoken aloud when due
- 📈 Habit tracking with streaks
- 🍅 Built-in Pomodoro focus timer
- 🗓️ Daily proactive briefing generated automatically from stored memory
- 🌐 Deployed as a web app via Streamlit Community Cloud

## Tech stack
- Python, Streamlit
- Groq API (LLM inference)
- Edge-TTS (voice synthesis)
- Local JSON storage for memory/tasks/habits/reminders

## Live demo
[shaneai-jvwlr57fkr462whshrfkuq.streamlit.app](https://shaneai-jvwlr57fkr462whshrfkuq.streamlit.app)

## Setup
```bash
git clone https://github.com/subhrajeetshaw008-bit/Shane.ai
cd Shane.ai
pip install -r requirements.txt
streamlit run app.py
```
You'll need a free [Groq API key](https://console.groq.com) added to Streamlit Secrets as `GROQ_API_KEY`.

## Status
Actively developed — see commit history for latest features.
