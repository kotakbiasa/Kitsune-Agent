# Kitsune Memory

Dokumen ini menjelaskan cara memori Kitsune bekerja di runtime Telegram.

## Storage

Kitsune memakai beberapa storage:

- `memory_db/`: database ChromaDB untuk long-term memory.
- `data/memory.md`: journal Markdown otomatis untuk bahan review dan improvement bot.
- `data/learned_preferences.json`: profil ringan per user, seperti jumlah interaksi dan kategori task yang sering dipakai.
- `data/approved_groups.json`: daftar grup Telegram yang sudah di-approve owner.
- `data/uploads/`: file Telegram yang diunduh untuk dibaca/diringkas.

`data/allowed_users.json` sudah legacy dan tidak dipakai untuk akses user. Akses user sekarang berbasis `OWNER_USER_IDS`, sedangkan akses grup berbasis `APPROVED_GROUP_IDS` atau approval runtime.

## Access Model

Mode akses saat ini:

- Private chat: hanya owner di `OWNER_USER_IDS`.
- Grup: owner bisa approve grup dengan `/approve` atau `/approve_group`.
- Setelah grup approved, member grup bisa chat dengan bot di grup itu.
- Command sensitif tetap owner-only:
  - `/tool`
  - `/sendfile`
  - `/access`
  - `/approve_group`
  - `/revoke_group`
  - `/improve`

## What Kitsune Learns

Kitsune menyimpan memori dari:

- Chat biasa, jika ada fakta/preferensi tahan lama.
- `/teach <fakta/preferensi>`.
- File teks yang dikirim ke Telegram, jika tidak terlihat berisi secret.
- Statistik routing model, untuk memilih model yang lebih cocok per kategori task.

Contoh memori yang layak disimpan:

- Nama atau panggilan user.
- Preferensi gaya jawaban.
- Preferensi bahasa.
- Koreksi stabil dari user.
- Ringkasan penting dari file teks yang memang relevan.

Contoh yang tidak seharusnya disimpan:

- Request satu kali.
- Token, password, API key, private key.
- Data sensitif dari `.env`, key/cert, atau file runtime.

## Auto Learning Config

Config utama di `.env`:

```env
AUTO_LEARNING=true
AUTO_MEMORY_EXTRACTION_EVERY=1
AUTO_LEARN_FROM_FILES=true
FILE_CONTEXT_MAX_CHARS=5000
TELEGRAM_FILE_READ_MAX_MB=10
AUTO_MEMORY_MARKDOWN=true
MEMORY_MARKDOWN_PATH=
```

Makna:

- `AUTO_LEARNING=true`: aktifkan penyimpanan fakta/preferensi otomatis.
- `AUTO_MEMORY_EXTRACTION_EVERY=1`: ekstraksi memori LLM dilakukan setiap interaksi sukses.
- `AUTO_LEARN_FROM_FILES=true`: file teks yang dikirim ke Telegram bisa masuk memori.
- `FILE_CONTEXT_MAX_CHARS=5000`: batas isi file yang dikirim ke model sebagai konteks.
- `TELEGRAM_FILE_READ_MAX_MB=10`: batas ukuran file Telegram yang dibaca.
- `AUTO_MEMORY_MARKDOWN=true`: append journal human-readable setiap ada interaction, learned memory, dan improvement signal.
- `MEMORY_MARKDOWN_PATH=`: override lokasi journal. Jika kosong, default ke `data/memory.md`.

## Improvement Journal

`data/memory.md` dibuat otomatis sebagai bahan improvement bot. Isinya bukan sumber utama memori runtime, tapi catatan review yang mudah dibaca:

- interaction singkat, termasuk task/model/success heuristic.
- learned memory dari direct rule, LLM extraction, `/teach`, dan file Telegram.
- improvement signal jika jawaban terlihat gagal/lemah.

Sebelum ditulis, konten akan direduksi untuk pola secret umum seperti `token`, `password`, `api_key`, dan private key block. Tetap hindari mengirim secret ke bot.

## Runtime Commands

- `/memory`: melihat memori terbaru user.
- `/teach <teks>`: menyimpan memori eksplisit.
- `/forget`: menghapus semua memori user.
- `/reset`: menghapus chat history sementara, bukan long-term memory.
- `/access`: melihat owner dan approved groups.

## Safety Rules

Kitsune tidak boleh mengubah source code production otomatis hanya karena learning. Self-improvement hanya membuat proposal reviewable lewat `/improve`.

File yang ditolak untuk dikirim/dibaca sebagai local tool:

- `.env`
- `.venv`
- `memory_db`
- `__pycache__`
- `*.pem`
- `*.key`
- `*.p12`
- `*.pfx`
- private key umum seperti `id_rsa` dan `id_ed25519`

Auto-learning dari file juga menolak konten yang terlihat mengandung:

- `api_key`
- `secret`
- `token`
- `password`
- private key block

## Maintenance

Untuk reset penuh memori:

1. Stop bot.
2. Backup jika perlu:
   - `memory_db/`
   - `data/learned_preferences.json`
3. Hapus `memory_db/` untuk reset vector memory.
4. Hapus atau kosongkan `data/learned_preferences.json` untuk reset profil ringan.
5. Start bot lagi.

Untuk approve grup permanen tanpa command Telegram:

```env
APPROVED_GROUP_IDS=-1001234567890,-1009876543210
```

Lalu restart bot.
