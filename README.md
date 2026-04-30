# 🦊 Kitsune-Agent

AI Telegram Agent yang terus belajar dan berkembang — dengan multi-model routing, memori jangka panjang, web search, dan self-learning.

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## ✨ Fitur

| Fitur | Deskripsi |
|-------|-----------|
| 🧠 **Multi-Model Routing** | Otomatis pilih model AI terbaik (Ollama, OpenAI, Anthropic, Gemini, OpenRouter) berdasarkan jenis tugas |
| 💾 **Memori Jangka Panjang** | Ingat fakta, preferensi, dan identitas user via ChromaDB |
| 🔍 **Web Search** | Cari info real-time via Ollama Web Search API & DuckDuckGo |
| ⏰ **Pengingat** | Buat pengingat dengan `/remind` |
| 👍 **Feedback Loop** | Tombol 👍/👎 setelah setiap jawaban untuk self-improvement |
| 🖥️ **Terminal** | Jalankan perintah shell langsung dari Telegram (owner only) |
| ⚙️ **Config Live** | Edit konfigurasi bot tanpa restart |
| 📁 **File Reader** | Upload file Telegram untuk dianalisis |
| 🔄 **Self-Learning** | Bot belajar dari setiap interaksi dan optimize routing |
| 🔐 **Access Control** | Owner-only + approved groups |

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/kotakbiasa/Kitsune-Agent.git
cd Kitsune-Agent
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Konfigurasi `.env`

```env
TELEGRAM_BOT_TOKEN=your_bot_token_from_botfather
OWNER_USER_IDS=123456789

# LLM API Keys (pilih salah satu atau semua)
OLLAMA_API_KEY=your_ollama_key
OPENROUTER_API_KEY=your_openrouter_key
ANTHROPIC_API_KEY=your_anthropic_key
GEMINI_API_KEY=your_gemini_key

# Web Search
ENABLE_WEB_SEARCH=true

# Optional
ENABLE_STREAMING=true
ENABLE_SELF_IMPROVE=true
ENABLE_LOCAL_TOOLS=true
ENABLE_SHELL_TOOL=false
```

### 3. Jalankan

```bash
python -m kitsune
```

## 📱 Command List

| Command | Deskripsi | Akses |
|---------|-----------|-------|
| `/start` | Mulai percakapan | Semua |
| `/help` | Daftar perintah | Semua |
| `/whoami` | Lihat profil & statistik | Semua |
| `/intro` | Isi profil (nama, job, minat) | Semua |
| `/search <query>` | Cari di web | Semua |
| `/remind <waktu> <pesan>` | Buat pengingat | Semua |
| `/reminders` | Lihat pengingat aktif | Semua |
| `/cancel_reminder <id>` | Batalkan pengingat | Semua |
| `/memory` | Lihat memori yang dipelajari | Semua |
| `/teach <fakta>` | Ajari bot sesuatu | Semua |
| `/forget` | Hapus semua memori | Semua |
| `/reset` | Reset percakapan | Semua |
| `/stats` | Statistik penggunaan | Semua |
| `/model` | Info model & routing | Semua |
| `/config` | Lihat/edit konfigurasi | Owner |
| `/terminal <cmd>` | Jalankan shell | Owner |
| `/tool <nama> <args>` | Jalankan tool lokal | Owner |
| `/sendfile <path>` | Kirim file ke chat | Owner |
| `/access` | Status akses | Owner |
| `/improve <ide>` | Buat proposal improvement | Owner |
| `/improvements` | Lihat proposal | Owner |

## 🧠 Auto-Learning

Bot otomatis menangkap informasi dari obrolan:

- **Nama** — `"Nama saya Budi"` → bot catat
- **Pekerjaan** — `"Saya developer"` → bot catat
- **Minat** — `"Aku suka Python & AI"` → bot catat
- **Panggilan** — `"Panggil aku mas"` → bot catat

Tidak perlu command `/intro`! Cukup ngobrol biasa. 🧠

## 🔍 Web Search

Tanya langsung tanpa command:
- `"Berita AI terbaru apa?"` → auto search → jawaban up-to-date
- `"Harga bitcoin hari ini"` → auto search → jawaban real-time

Atau pakai command eksplisit:
- `/search harga bitcoin`
- `/search berita teknologi`

Provider: **Ollama Web Search API** (primary) → DuckDuckGo (fallback) → Google (fallback)

## ⚙️ Arsitektur

```
User Message → Router (classify task) → Brain (LLM call) → Telegram
                      ↓
              Memory (ChromaDB) ← Learner (extract knowledge)
                      ↓
              Web Search (real-time info)
```

**Komponen:**
- `bot.py` — Telegram handlers & orchestration
- `router.py` — Task classifier & model switcher
- `brain.py` — LLM interface via LiteLLM
- `memory.py` — ChromaDB long-term memory
- `learner.py` — Self-learning engine
- `search.py` — Web search (Ollama API + DuckDuckGo)
- `reminder.py` — Scheduler & reminder system
- `tools.py` — Local tool registry

## 🛡️ Keamanan

- Owner-only access by default
- Shell command terproteksi (`ENABLE_SHELL_TOOL=false` default)
- Path traversal protection
- Sensitive file redaction (API keys, tokens)
- Circuit breaker untuk model yang gagal
- Feedback button hanya bisa diklik oleh user yang bersangkutan

## 📄 License

MIT License — lihat [LICENSE](LICENSE) untuk detail.

---

<div align="center">

**🦊 Kitsune-Agent** — *AI yang terus belajar dari setiap obrolan*

</div>
