# 🦊 Kitsune-Agent

Kitsune-Agent adalah AI agent untuk Telegram yang bisa **belajar sendiri** dari interaksi dan **otomatis ganti model (auto-switch)** berdasarkan jenis task yang dikerjakan.

## ✨ Fitur Utama

- **🧠 Auto Model Routing:** Kitsune memakai pool model Ollama Cloud per kategori tugas, lalu otomatis mencoba model berikutnya kalau model utama gagal atau tidak tersedia.
- **🗄️ Self-Learning Memory:** Setiap kali kamu ngobrol, Kitsune menyimpan fakta dan preferensimu di ChromaDB. Embedding default memakai hash ringan bawaan Python, jadi tidak perlu download model lokal di cloud.
- **🎓 Explicit Teaching:** Kamu bisa mengajari Kitsune secara langsung dengan `/teach`, lalu mengecek memori yang tersimpan lewat `/memory`.
- **🔄 Auto-Optimization:** Setelah beberapa kali interaksi, Kitsune menganalisis model mana yang sering gagal atau lambat, lalu menyesuaikan *routing rules*-nya secara otomatis.
- **🧩 Safe Self-Improve:** Kitsune bisa membuat proposal patch dari `/improve`, tetapi tidak mengubah kode production otomatis.

## 🚀 Cara Install

### 1. Clone & Install Dependencies
Pastikan kamu menggunakan Python 3.10+.
```bash
git clone <repository-url>
cd Kitsune-Agent
uv sync
```

Jika belum memakai `uv`, instal dependency tetap bisa lewat pip:
```bash
pip install -r requirements.txt
```

### 2. Konfigurasi API Keys
Salin file template `.env`:
```bash
cp .env.example .env
```
Buka file `.env` dan masukkan API keys yang dibutuhkan:
- **`TELEGRAM_BOT_TOKEN`**: Dapatkan dari [@BotFather](https://t.me/BotFather) di Telegram.
- **LLM API Key**: Untuk Ollama Cloud, isi `OLLAMA_API_KEY`. Fallback utama memakai pool model Ollama; provider lain opsional.
- **Cloud default:** Ollama memakai hosted API `https://ollama.com`, bukan local `localhost`.

### 3. Jalankan Bot
```bash
uv run python main.py
```

## ☁️ Deploy Cloud / Docker

Kitsune sekarang Ollama Cloud-first: default routing memakai hosted API `https://ollama.com`, tidak mengarah ke `localhost`, dan tidak membutuhkan local embedding model.

Build dan jalankan:
```bash
docker build -t kitsune-agent .
docker run -d --name kitsune-agent \
  --env-file .env \
  -v kitsune-data:/app/data \
  -v kitsune-memory:/app/memory_db \
  kitsune-agent
```

Minimal `.env` untuk cloud:
```env
TELEGRAM_BOT_TOKEN=123456:telegram_token
OLLAMA_API_KEY=ollama_...
OLLAMA_API_BASE=https://ollama.com
OLLAMA_FAST_MODEL=ministral-3:3b-cloud
OLLAMA_REASONING_MODEL=deepseek-v4-flash:cloud
OLLAMA_CODING_MODEL=qwen3-coder-next:cloud
OLLAMA_PRODUCTIVITY_MODEL=minimax-m2.7:cloud
OLLAMA_DEFAULT_MODEL=ministral-3:3b-cloud
OLLAMA_AUTO_DISCOVER_MODELS=true
OLLAMA_MODEL_POOL=ministral-3:3b-cloud,rnj-1:8b-cloud,ministral-3:8b-cloud,ministral-3:14b-cloud,qwen3-coder-next:cloud,qwen3.5:cloud,gemini-3-flash-preview:cloud,nemotron-3-nano:30b-cloud,devstral-small-2:24b-cloud,gemma4:31b-cloud,minimax-m2.7:cloud,minimax-m2.5:cloud,deepseek-v4-flash:cloud,nemotron-3-super:cloud,glm-5.1:cloud,glm-5:cloud,kimi-k2.6:cloud,kimi-k2.5:cloud,qwen3-next:80b-cloud,devstral-2:123b-cloud,deepseek-v4-pro:cloud,qwen3.5:397b-cloud,cogito-2.1:671b-cloud
OLLAMA_SIMPLE_MODELS=ministral-3:3b-cloud,rnj-1:8b-cloud,ministral-3:8b-cloud,qwen3.5:cloud,gemini-3-flash-preview:cloud
OLLAMA_REASONING_MODELS=deepseek-v4-flash:cloud,nemotron-3-super:cloud,glm-5.1:cloud,gemini-3-flash-preview:cloud,kimi-k2.6:cloud
OLLAMA_CODING_MODELS=qwen3-coder-next:cloud,rnj-1:8b-cloud,devstral-small-2:24b-cloud,glm-5.1:cloud,minimax-m2.7:cloud,deepseek-v4-flash:cloud,kimi-k2.6:cloud
OLLAMA_CREATIVE_MODELS=qwen3.5:cloud,gemini-3-flash-preview:cloud,minimax-m2.7:cloud,kimi-k2.6:cloud
OLLAMA_MATH_MODELS=deepseek-v4-flash:cloud,nemotron-3-super:cloud,glm-5.1:cloud,kimi-k2.6:cloud
OLLAMA_TRANSLATION_MODELS=ministral-3:3b-cloud,qwen3.5:cloud,gemini-3-flash-preview:cloud,ministral-3:14b-cloud
OLLAMA_SUMMARIZATION_MODELS=qwen3.5:cloud,gemini-3-flash-preview:cloud,minimax-m2.7:cloud,deepseek-v4-flash:cloud
OPENROUTER_API_KEY=
OWNER_USER_IDS=123456789
APPROVED_GROUP_IDS=
ENABLE_LOCAL_TOOLS=false
ENABLE_SHELL_TOOL=false
TELEGRAM_CONNECT_TIMEOUT=30
TELEGRAM_READ_TIMEOUT=60
TELEGRAM_WRITE_TIMEOUT=60
TELEGRAM_POOL_TIMEOUT=30
TELEGRAM_FILE_READ_MAX_MB=10
FAST_ROUTING=true
BACKGROUND_LEARNING=true
AUTO_LEARNING=true
AUTO_MEMORY_EXTRACTION_EVERY=1
AUTO_LEARN_FROM_FILES=true
FILE_CONTEXT_MAX_CHARS=5000
AUTO_MEMORY_MARKDOWN=true
MEMORY_MARKDOWN_PATH=
ENABLE_SELF_IMPROVE=true
ENABLE_STREAMING=true
TELEGRAM_STREAM_MODE=draft
STREAM_EDIT_INTERVAL=1
STREAM_MIN_CHARS=12
TELEGRAM_CHAT_ACTION_INTERVAL=4
```

`STREAM_EDIT_INTERVAL` dan `STREAM_MIN_CHARS` punya batas bawah aman di kode: minimal 1 detik dan 12 karakter per update. `TELEGRAM_CHAT_ACTION_INTERVAL=4` menahan action seperti `typing`, `upload_document`, dan `upload_photo` agar tidak dikirim beruntun terlalu cepat.

`AUTO_LEARNING=true` membuat Kitsune menyimpan fakta/preferensi tahan lama dari percakapan tanpa konfirmasi. `AUTO_LEARN_FROM_FILES=true` membuat file teks yang dikirim ke Telegram ikut masuk memori jika tidak terlihat berisi secret seperti token/password/private key. `AUTO_MEMORY_MARKDOWN=true` membuat journal human-readable di `data/memory.md` sebagai bahan review dan improvement bot. Ini tidak membuat Kitsune mengubah source code production otomatis.

Ollama Cloud routing memakai pool multi-model. `OLLAMA_AUTO_DISCOVER_MODELS=true` mencoba membaca daftar model yang tersedia dari akun Ollama; jika discovery gagal, Kitsune tetap memakai `OLLAMA_MODEL_POOL` dan pool per kategori. Setiap request mendapat kandidat model berurutan, lalu Brain mencoba model berikutnya otomatis kalau model sebelumnya gagal.

Default Ollama Cloud routing:
- `ministral-3:3b-cloud` / `rnj-1:8b-cloud`: simple QA dan request harian yang hemat usage
- `qwen3-coder-next:cloud`: coding default yang efisien sebelum fallback ke Devstral/GLM/Kimi
- `deepseek-v4-flash:cloud`: reasoning/math default sebelum fallback ke model frontier yang lebih berat
- `qwen3.5:cloud` / `gemini-3-flash-preview:cloud`: translation, summarization, dan creative work

Streaming aktif secara default. `TELEGRAM_STREAM_MODE=draft` memakai Bot API `sendMessageDraft` untuk private chat. `STREAM_EDIT_INTERVAL=1` menjaga update tetap terasa live tanpa memukul flood limit Telegram. Kalau draft tidak tersedia, bot fallback ke edit-message streaming. Jika Telegram mulai rate limit, naikkan `STREAM_EDIT_INTERVAL` agar update tidak terlalu sering.

Untuk respons cepat, `FAST_ROUTING=true` menghindari panggilan LLM khusus hanya untuk klasifikasi, dan `BACKGROUND_LEARNING=true` membuat proses penyimpanan memori berjalan setelah jawaban dikirim.

Kalau deploy tanpa Docker:
```bash
uv sync --frozen
uv run python main.py
```

## 🎮 Cara Penggunaan (Telegram)

Buka Telegram, cari bot-mu, lalu klik **Start**. 
Beberapa perintah yang tersedia:
- `/start` — Memulai percakapan
- `/help` — Menampilkan bantuan
- `/stats` — Melihat jumlah interaksimu dan statistik bot
- `/model` — Melihat status model routing (model apa dipakai untuk tugas apa)
- `/memory` — Melihat memori jangka panjang yang sudah dipelajari Kitsune
- `/teach <hal penting>` — Mengajari Kitsune fakta atau preferensi penting secara eksplisit
- `/tools` — Melihat tool lokal yang tersedia jika `ENABLE_LOCAL_TOOLS=true`
- `/tool <nama> <argumen>` — Menjalankan tool lokal eksplisit seperti `read_file`, `list_files`, atau `grep`
- `/sendfile <path> [caption]` — Mengirim file dari workspace Kitsune ke chat Telegram saat ini
- Kirim document/photo langsung ke bot untuk deteksi jenis file dan preview teks jika formatnya terbaca
- `/approve` atau `/approve_group` — Owner approve grup tempat command dikirim
- `/deny` atau `/revoke_group` — Owner mencabut approval grup tempat command dikirim
- `/access` — Owner melihat status mode owner-only dan daftar owner dari `.env`.
- `/improve <ide/masalah>` — Membuat proposal patch aman di `data/improvement_proposals/`
- `/improvements` — Melihat proposal improvement terbaru
- `/reset` — Menghapus memori jangka pendek percakapan saat ini
- `/forget` — Menghapus **semua** memori jangka panjang tentangmu

Kitsune memakai mode owner-only untuk user pribadi dan command sensitif. Di grup, owner bisa menjalankan `/approve` atau `/approve_group` agar bot merespons member grup tersebut. Approval runtime disimpan di `data/approved_groups.json`; grup permanen juga bisa diisi lewat `APPROVED_GROUP_IDS` di `.env`.

## 🛠️ Local Tools

Kitsune bisa memakai tool lokal opsional yang dibuat sebagai subset aman untuk Telegram.

Aktifkan di `.env`:
```bash
OWNER_USER_IDS=123456789
APPROVED_GROUP_IDS=
ENABLE_LOCAL_TOOLS=true
ENABLE_SHELL_TOOL=false
MAX_SEND_FILE_MB=45
```

Contoh:
```text
/tools
/tool list_files . *.py
/tool read_file kitsune/bot.py 0 80
/tool file_info README.md
/tool grep memory kitsune *.py
/tool send_file data/report.txt "hasil audit"
/tool send_photo data/screenshot.png "preview"
/sendfile data/report.txt "hasil audit"
/tool source_tools Bash
/tool source_commands agents
/tool route "read file and grep memory"
/tool claude FileReadTool kitsune/bot.py 0 80
/tool claude SendMessageTool file data/report.txt "hasil audit"
```

`file_info` mendeteksi jenis file, MIME, ukuran, dan preview untuk file teks. File yang dikirim langsung ke Telegram disimpan di `data/uploads/` secara default dan dibatasi oleh `TELEGRAM_FILE_READ_MAX_MB`. Kalau document teks/code/JSON/CSV/Markdown dikirim dengan caption, Kitsune memakai isi file sebagai konteks jawaban; tanpa caption, Kitsune tetap menjelaskan jenis dan ringkasan isi file.

`source_tools`, `source_commands`, `source_search`, dan `route` memakai source/metadata dari `kitsune/src/` yang sudah ikut repo. Beberapa tool source yang aman dipetakan ke implementasi lokal lewat `/tool claude`, misalnya `FileReadTool`, `GlobTool`, dan `BashTool`.

`send_file`, `send_photo`, dan `/sendfile` hanya bisa mengirim file yang berada di dalam workspace Kitsune-Agent. File secret/runtime seperti `.env`, `.venv`, `memory_db`, key/cert, dan `__pycache__` ditolak. Ukuran maksimal default 45 MB dan bisa diatur lewat `MAX_SEND_FILE_MB`.

`/tool shell` dan `/tool claude BashTool` sengaja nonaktif secara default. Aktifkan hanya kalau bot tidak publik dan `OWNER_USER_IDS` sudah diisi.

## 📁 Struktur Direktori

- `main.py`: Entry point aplikasi.
- `kitsune/bot.py`: Handler bot Telegram.
- `kitsune/router.py`: Logika klasifikasi tugas dan routing model.
- `kitsune/brain.py`: Interface ke LLM (menggunakan LiteLLM).
- `kitsune/memory.py`: Sistem memori jangka panjang berbasis ChromaDB.
- `kitsune/learner.py`: Engine yang mengekstrak pengetahuan dan melakukan optimasi.
- `kitsune/self_improve.py`: Proposal self-improvement yang aman untuk review manual.
- `kitsune/tools.py`: Registry tool lokal aman untuk `/tools` dan `/tool`.
- `kitsune/src/`: Source/metadata tool yang ikut repo agar jalan di Docker/VPS tanpa path lokal.
- `data/`: Folder yang menyimpan *routing rules* dan preferensi (akan terbuat otomatis).
- `memory_db/`: Folder penyimpanan database vektor ChromaDB (akan terbuat otomatis).
