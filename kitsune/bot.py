"""
Kitsune Telegram Bot using aiogram.
Connects all components: Router, Brain, Memory, Learner, and local tools.
"""

from __future__ import annotations

import asyncio
import html
import logging
import re
import subprocess
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from aiogram import Bot, Dispatcher, F, Router as AiogramRouter
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums import ChatAction, ChatType, ParseMode
from aiogram.exceptions import TelegramBadRequest, TelegramNetworkError, TelegramRetryAfter
from aiogram.filters import Command
from aiogram.methods import SendMessageDraft
from aiogram.types import CallbackQuery, FSInputFile, InlineKeyboardButton, InlineKeyboardMarkup, Message

from kitsune.backup import MemoryBackup
from kitsune.brain import Brain, BrainResponse
from kitsune.config import Config
from kitsune.health import HealthMonitor
from kitsune.learner import Learner
from kitsune.memory import MemorySystem
from kitsune.model_utils import pick_fast_model, resolve_model_alias
from kitsune.reminder import ReminderSystem
from kitsune.router import Router
from kitsune.search import WebSearch, SearchTool
from kitsune.self_improve import SelfImprover
from kitsune.tools import (
    LocalToolRegistry,
    PreparedFile,
    format_size,
    inspect_file,
    shell_enabled_from_env,
    tools_enabled_from_env,
)

logger = logging.getLogger("kitsune.bot")


@dataclass
class _FeedbackMeta:
    interaction_id: str
    user_id: int
    user_message: str
    bot_response: str
    task_category: str
    model_used: str
    responded: bool = False


class KitsuneBot:
    """Main Telegram bot class that orchestrates all components."""

    def __init__(self, config: Config):
        self.config = config
        self.memory = MemorySystem()
        self.brain = Brain(config)
        self.router = Router(config)
        self.learner = Learner(config, self.memory, self.brain, self.router)
        self.self_improver = SelfImprover(config, self.brain) if config.enable_self_improve else None
        self.local_tools = (
            LocalToolRegistry(enable_shell=shell_enabled_from_env())
            if tools_enabled_from_env()
            else None
        )
        self.web_search = (
            WebSearch(
                ollama_api_key=config.ollama_api_key,
                ollama_api_base=config.ollama_api_base,
                google_api_key=config.google_api_key,
                google_cx=config.google_cx,
                max_results=config.web_search_max_results,
                timeout=config.web_search_timeout,
            )
            if config.enable_web_search
            else None
        )
        self.search_tool = SearchTool(self.web_search)

        self.reminders = ReminderSystem(
            storage_path=self.config.data_dir / "reminders.json",
            callback=self._on_reminder_trigger,
        )

        self.health_monitor = HealthMonitor(config, router=self.router)
        self.backup = MemoryBackup(self.memory, self.config.data_dir)

        self._chat_history: dict[tuple[int, int], list[dict]] = {}
        self._draft_retry_until: dict[int, float] = {}
        self._last_chat_action_at: dict[tuple[int, str], float] = {}
        self._feedback_registry: dict[int, _FeedbackMeta] = {}

        # Implicit negative feedback tracking
        self._last_user_message: dict[int, str] = {}  # user_id -> last message text
        self._last_response_meta: dict[int, dict] = {}  # user_id -> {model, category, response_text, timestamp}

        # User-requested model overrides (e.g., "gunakan kimi" → override routing for this user)
        self._user_model_override: dict[int, str] = {}  # user_id -> canonical model string

        self._implicit_negative_keywords = (
            r"\b(salah|ngawur|gak jelas|tidak jelas|ngaco|bodoh|stupid|nonsense|wrong|incorrect|bad|terrible|awful|useless|ga bisa|gak bisa|tidak bisa|gagal|failed|rusak|broken)\b",
            r"\b(bukan itu|bukan gitu|not that|not what i|bukan yang|kurang|kurang tepat|kurang baik|not good|not helpful|tidak membantu)\b",
            r"\b(ulangi|repeat|coba lagi|try again|sekali lagi|once more)\b",
            r"\b(apa ini|what is this|apa maksud|what do you mean|ngomong apa|talking about)\b",
        )

        self.bot: Bot | None = None
        self.dp: Dispatcher | None = None
        self._bot_username: str | None = None
        self._bot_id: int | None = None

    def run(self):
        """Start aiogram polling."""
        asyncio.run(self._run())

    async def _run(self):
        logger.info("🦊 Starting Kitsune Bot with aiogram...")

        session = AiohttpSession(timeout=self.config.telegram_read_timeout)
        self.bot = Bot(
            token=self.config.telegram_token,
            session=session,
            default=DefaultBotProperties(parse_mode=None),
        )
        self.dp = Dispatcher()

        # Fetch bot info for mention detection
        try:
            bot_info = await self.bot.get_me()
            self._bot_username = bot_info.username
            self._bot_id = bot_info.id
            logger.info("🤖 Bot @%s (id=%d) ready", self._bot_username, self._bot_id)
        except Exception as e:
            logger.warning("Could not fetch bot info: %s", e)
            self._bot_username = None
            self._bot_id = None

        router = AiogramRouter()
        self._register_handlers(router)
        self.dp.include_router(router)

        self.reminders.start()
        logger.info("⏰ Reminder scheduler started.")

        self.health_monitor.start()
        logger.info("🏥 Health monitor started.")

        if self.config.enable_auto_backup:
            await self.backup.start(self.config.auto_backup_interval_hours)
            logger.info("💾 Auto-backup started (every %.1fh).", self.config.auto_backup_interval_hours)

        logger.info("🚀 Kitsune Bot is ready! Listening for messages...")
        await self.dp.start_polling(
            self.bot,
            polling_timeout=int(self.config.telegram_read_timeout),
            allowed_updates=self.dp.resolve_used_update_types(),
        )

    def _register_handlers(self, router: AiogramRouter):
        router.message.register(self._cmd_start, Command("start"))
        router.message.register(self._cmd_help, Command("help"))
        router.message.register(self._cmd_stats, Command("stats"))
        router.message.register(self._cmd_model, Command("model"))
        router.message.register(self._cmd_memory, Command("memory"))
        router.message.register(self._cmd_teach, Command("teach"))
        router.message.register(self._cmd_tools, Command("tools"))
        router.message.register(self._cmd_tool, Command("tool"))
        router.message.register(self._cmd_sendfile, Command("sendfile", "file"))
        router.message.register(self._cmd_approve_group, Command("approve_group", "approvegroup"))
        router.message.register(self._cmd_revoke_group, Command("revoke_group", "deny_group", "revokegroup"))
        router.message.register(self._cmd_approve_or_notice, Command("allow", "approve", "deny"))
        router.message.register(self._cmd_access, Command("access"))
        router.message.register(self._cmd_improve, Command("improve"))
        router.message.register(self._cmd_improvements, Command("improvements"))
        router.message.register(self._cmd_health, Command("health"))
        router.message.register(self._cmd_backup, Command("backup"))
        router.message.register(self._cmd_backups, Command("backups"))
        router.message.register(self._cmd_whoami, Command("whoami"))
        router.message.register(self._cmd_intro, Command("intro"))
        router.message.register(self._cmd_config, Command("config"))
        router.message.register(self._cmd_terminal, Command("terminal", "sh", "shell"))
        router.message.register(self._cmd_search, Command("search"))
        router.message.register(self._cmd_remind, Command("remind"))
        router.message.register(self._cmd_reminders, Command("reminders"))
        router.message.register(self._cmd_cancel_reminder, Command("cancel_reminder"))
        router.message.register(self._cmd_forget, Command("forget"))
        router.message.register(self._cmd_reset, Command("reset"))
        router.message.register(self._handle_document_message, F.document)
        router.message.register(self._handle_photo_message, F.photo)
        router.message.register(self._handle_sticker, F.sticker)
        router.message.register(self._handle_unsupported_media, F.video | F.voice | F.audio | F.animation)
        # Group mention/reply handlers FIRST so they take priority over _handle_message
        router.message.register(
            self._handle_group_mention,
            F.text,
            lambda message: bool(
                message.text
                and message.chat.type in {ChatType.GROUP, ChatType.SUPERGROUP}
            ),
        )
        router.message.register(
            self._handle_message,
            F.text,
            lambda message: bool(
                message.text
                and not message.text.startswith("/")
                and message.chat.type == ChatType.PRIVATE
            ),
        )
        router.callback_query.register(self._feedback_callback)
        router.errors.register(self._error_handler)

    def _is_authorized(self, user_id: int) -> bool:
        return self._is_owner(user_id)

    def _is_owner(self, user_id: int) -> bool:
        return self.config.is_owner(user_id)

    def _is_approved_group_message(self, message: Message) -> bool:
        return message.chat.type in {ChatType.GROUP, ChatType.SUPERGROUP} and self.config.is_group_approved(message.chat.id)

    def _is_message_authorized(self, message: Message) -> bool:
        user = message.from_user
        return bool(user and (self._is_owner(user.id) or self._is_approved_group_message(message)))

    def _is_owner_message(self, message: Message) -> bool:
        user = message.from_user
        return bool(user and self._is_owner(user.id))

    def _history_key(self, message: Message) -> tuple[int, int]:
        user_id = message.from_user.id if message.from_user else message.chat.id
        return (message.chat.id, user_id)

    async def _cmd_start(self, message: Message):
        user = message.from_user
        if not user or not self._is_message_authorized(message):
            await self._safe_answer(message, "⛔ Maaf, kamu tidak punya akses ke bot ini.")
            return

        welcome = (
            f"Halo {user.first_name}! 🦊\n\n"
            f"Aku **Kitsune**, AI agent yang terus belajar dan berkembang!\n\n"
            f"✨ **Kemampuanku:**\n"
            f"• Menjawab pertanyaan apapun\n"
            f"• Menulis & debug kode\n"
            f"• Membaca jenis file dan preview file teks\n"
            f"• Menulis cerita & konten kreatif\n"
            f"• Menerjemahkan bahasa\n"
            f"• Menyelesaikan soal matematika\n"
            f"• Merangkum teks panjang\n\n"
            f"🧠 Aku otomatis memilih model AI terbaik untuk setiap tugas, "
            f"dan belajar dari setiap interaksi kita!\n\n"
            f"💡 **Mau aku kenali kamu?**\n"
            f"Bilang aja langsung: _nama saya Budi, developer fullstack, suka Python_.\n"
            f"Aku bakal otomatis catat! 🧠\n\n"
            f"Ketik /help untuk melihat semua perintah."
        )
        await self._safe_answer(message, welcome, parse_mode=ParseMode.MARKDOWN)

    async def _cmd_help(self, message: Message):
        if not self._authorized_message(message):
            return

        help_text = (
            "🦊 **Perintah Kitsune:**\n\n"
            "/start — Mulai percakapan\n"
            "/help — Tampilkan bantuan ini\n"
            "/stats — Lihat statistik & performamu\n"
            "/model — Lihat info model & routing\n"
            "/memory — Lihat memori yang dipelajari\n"
            "/teach <hal penting> — Ajari aku preferensi/fakta secara eksplisit\n"
            "/tools — Lihat tool lokal yang tersedia\n"
            "/tool <nama> <argumen> — Jalankan tool lokal eksplisit\n"
            "/sendfile <path> [caption] — Kirim file workspace ke chat ini\n"
            "/access — Owner melihat status akses\n"
            "/whoami — Lihat profil & statistik pribadimu\n"
            "/intro — Kenalkan dirimu (nama, pekerjaan, minat, dll)\n"
            "/remind <waktu> <pesan> — Buat pengingat\n"
            "/reminders — Lihat pengingat aktif\n"
            "/cancel_reminder <id> — Batalkan pengingat\n"
            "/search <query> — Cari di web (real-time info)\n"
            "/config — Lihat/edit konfigurasi bot (owner only)\n"
            "/terminal <command> — Jalankan perintah shell (owner only)\n"
            "/improve <ide/masalah> — Buat proposal patch aman untuk direview\n"
            "/improvements — Lihat proposal improvement terbaru\n"
            "/forget — Hapus semua memori tentangmu\n"
            "/reset — Reset percakapan saat ini\n\n"
            "💡 **Tips:**\n"
            "• Langsung kirim pesan untuk mulai chat\n"
            "• Kirim document/photo untuk cek jenis file dan membaca preview teks\n"
            "• Bilang 'buatkan script Python untuk X' — bot otomatis generate & kirim file\n"
            "• Bilang 'gunakan deepseek' — ganti model secara langsung\n"
            "• Bilang 'jadi lebih santai' atau 'kamu terlalu formal' — otomatis ubah gaya bot"
        )
        await self._safe_answer(message, help_text, parse_mode=ParseMode.MARKDOWN)

    async def _cmd_stats(self, message: Message):
        user = message.from_user
        if not user or not self._is_message_authorized(message):
            return

        user_stats = self.learner.get_user_stats(user.id)
        mem_stats = self.memory.get_stats()
        brain_stats = self.brain.get_stats()
        stats = (
            f"{user_stats}\n\n"
            f"🗄️ **Memory:**\n"
            f"  Interactions stored: {mem_stats['total_interactions']}\n"
            f"  Knowledge items: {mem_stats['total_knowledge']}\n\n"
            f"🧠 **Brain (this session):**\n"
            f"  Tokens used: {brain_stats['total_tokens']:,}\n"
            f"  Cost: ${brain_stats['total_cost']:.4f}"
        )
        await self._safe_answer(message, stats, parse_mode=ParseMode.MARKDOWN)

    async def _cmd_model(self, message: Message):
        if not self._authorized_message(message):
            return
        await self._safe_answer(message, self.router.get_routing_stats(), parse_mode=ParseMode.MARKDOWN)

    async def _cmd_forget(self, message: Message):
        user = message.from_user
        if not user or not self._is_message_authorized(message):
            return

        deleted = self.memory.forget_user(user.id)
        self._chat_history.pop(self._history_key(message), None)
        await self._safe_answer(
            message,
            f"🗑️ Done! Saya sudah menghapus {deleted} memori tentangmu.\n"
            f"Kita mulai dari awal lagi! 🦊",
        )

    async def _cmd_memory(self, message: Message):
        user = message.from_user
        if not user or not self._is_message_authorized(message):
            return

        memories = self.memory.get_recent_knowledge(user.id, limit=12)
        if not memories:
            await self._safe_answer(message, "Belum ada memori jangka panjang. Pakai /teach atau lanjut ngobrol dulu.")
            return

        lines = ["🧠 Memori yang kupelajari:"]
        for idx, item in enumerate(memories, start=1):
            metadata = item.get("metadata", {})
            mem_type = metadata.get("type", "memory")
            source = metadata.get("source", "unknown")
            lines.append(f"{idx}. [{mem_type}] {item['document'][:220]} (source: {source})")
        await self._safe_answer(message, "\n".join(lines))

    async def _cmd_teach(self, message: Message):
        user = message.from_user
        if not user or not self._is_message_authorized(message):
            return

        fact = self._command_args(message)
        if not fact:
            await self._safe_answer(
                message,
                "Kirim seperti ini:\n/teach Panggil saya Fauzan dan jawab singkat kalau saya tanya teknis.",
            )
            return

        memory_id = self.learner.teach_user(user.id, fact)
        await self._safe_answer(
            message,
            "Siap, itu sudah kusimpan sebagai memori penting."
            if memory_id
            else "Aku belum bisa menyimpan memori itu. Coba ulangi nanti.",
        )

    async def _cmd_reset(self, message: Message):
        if not self._authorized_message(message):
            return
        self._chat_history.pop(self._history_key(message), None)
        await self._safe_answer(message, "🔄 Percakapan direset! Memori jangka panjang tetap tersimpan.")

    async def _cmd_tools(self, message: Message):
        if not self._is_owner_message(message):
            return
        if not self.local_tools:
            await self._safe_answer(message, "Tool lokal belum aktif. Set ENABLE_LOCAL_TOOLS=true di .env.")
            return
        await self._safe_answer(message, self.local_tools.describe())

    async def _cmd_tool(self, message: Message):
        await self._run_local_tool(message, self._command_args(message))

    async def _cmd_sendfile(self, message: Message):
        user = message.from_user
        if not user or not self._is_message_authorized(message):
            await self._safe_answer(message, "⛔ Maaf, kamu tidak punya akses ke bot ini.")
            return
        if not self.local_tools:
            await self._safe_answer(message, "🔧 Tool lokal belum aktif. Set ENABLE_LOCAL_TOOLS=true di .env.")
            return

        args = self._command_args(message)
        if not args:
            await self._safe_answer(message, "Kirim seperti ini:\n/sendfile data/report.txt caption opsional")
            return

        result = self.local_tools.run(f"send_file {args}")
        prefix = "✅" if result.ok else "❌"
        if not result.files:
            await self._safe_answer(message, f"{prefix}\n{result.output[:3500]}")
            return

        await self._safe_answer(message, f"{prefix}\n{result.output[:900]}")
        for prepared_file in result.files:
            await self._safe_send_prepared_file(message, prepared_file)

    async def _run_local_tool(self, message: Message, raw_args: str):
        if not self._is_owner_message(message):
            await self._safe_answer(message, "⛔ Hanya owner yang bisa pakai tool lokal.")
            return
        if not self.local_tools:
            await self._safe_answer(message, "🔧 Tool lokal belum aktif. Set ENABLE_LOCAL_TOOLS=true di .env.")
            return

        result = self.local_tools.run(raw_args)
        prefix = "OK" if result.ok else "ERROR"
        if not result.files:
            await self._safe_answer(message, f"{prefix}\n{result.output[:3500]}")
            return

        await self._safe_answer(message, f"{prefix}\n{result.output[:900]}")
        for prepared_file in result.files:
            await self._safe_send_prepared_file(message, prepared_file)

    async def _cmd_approve_group(self, message: Message):
        user = message.from_user
        if not user or not self._is_owner(user.id):
            return

        group_id = self._target_group_id(message)
        if group_id is None:
            await self._safe_answer(message, "Jalankan /approve_group di grup, atau pakai /approve_group <group_id>.")
            return

        added = self.config.add_approved_group(group_id)
        status = "diapprove" if added else "sudah approved"
        await self._safe_answer(message, f"OK, grup `{group_id}` {status}.", parse_mode=ParseMode.MARKDOWN)

    async def _cmd_revoke_group(self, message: Message):
        user = message.from_user
        if not user or not self._is_owner(user.id):
            return

        group_id = self._target_group_id(message)
        if group_id is None:
            await self._safe_answer(message, "Jalankan /revoke_group di grup, atau pakai /revoke_group <group_id>.")
            return

        removed = self.config.remove_approved_group(group_id)
        status = "dicabut" if removed else "tidak ada di daftar runtime"
        await self._safe_answer(message, f"OK, akses grup `{group_id}` {status}.", parse_mode=ParseMode.MARKDOWN)

    async def _cmd_approve_or_notice(self, message: Message):
        user = message.from_user
        if not user or not self._is_owner(user.id):
            return

        command = (message.text or "").split(maxsplit=1)[0].lower().lstrip("/")
        if message.chat.type in {ChatType.GROUP, ChatType.SUPERGROUP}:
            if command in {"allow", "approve"}:
                await self._cmd_approve_group(message)
                return
            if command == "deny":
                await self._cmd_revoke_group(message)
                return

        await self._safe_answer(
            message,
            "Akses user tetap owner-only. Untuk grup: jalankan /approve di grup untuk approve grup ini, atau /approve_group <group_id>.",
        )

    async def _cmd_access(self, message: Message):
        user = message.from_user
        if not user or not self._is_owner(user.id):
            return

        lines = [
            "Access status:",
            f"- owners: `{', '.join(str(uid) for uid in self.config.owner_user_ids) or '-'}`",
            "- user mode: `owner-only`",
            f"- env approved groups: `{', '.join(str(uid) for uid in self.config.approved_group_ids) or '-'}`",
            f"- runtime approved groups: `{', '.join(str(uid) for uid in self.config.runtime_groups.get('approved_group_ids', [])) or '-'}`",
        ]
        await self._safe_answer(message, "\n".join(lines), parse_mode=ParseMode.MARKDOWN)

    async def _cmd_improve(self, message: Message):
        user = message.from_user
        if not user or not self._is_owner(user.id):
            return
        if not self.self_improver:
            await self._safe_answer(message, "Self-improve belum aktif. Set ENABLE_SELF_IMPROVE=true di .env.")
            return

        request = self._command_args(message)
        if not request:
            await self._safe_answer(
                message,
                "Kirim seperti ini:\n/improve respon streaming kadang telat muncul di Telegram",
            )
            return

        await self._safe_send_chat_action(message)
        path = await self.self_improver.propose_from_request(user.id, request)
        await self._safe_answer(
            message,
            f"Proposal improvement sudah dibuat:\n`{path.relative_to(self.config.self_improve_dir.parent)}`\n\n"
            "Ini belum mengubah kode. Review file itu dulu sebelum patch diterapkan.",
            parse_mode=ParseMode.MARKDOWN,
        )

    async def _cmd_improvements(self, message: Message):
        if not self._is_owner_message(message):
            return
        if not self.self_improver:
            await self._safe_answer(message, "Self-improve belum aktif. Set ENABLE_SELF_IMPROVE=true di .env.")
            return

        proposals = self.self_improver.list_recent()
        if not proposals:
            await self._safe_answer(message, "Belum ada proposal improvement.")
            return

        base_dir = self.config.self_improve_dir.parent
        lines = ["Proposal improvement terbaru:"]
        for idx, path in enumerate(proposals, start=1):
            lines.append(f"{idx}. `{path.relative_to(base_dir)}`")
        await self._safe_answer(message, "\n".join(lines), parse_mode=ParseMode.MARKDOWN)

    async def _cmd_health(self, message: Message):
        if not self._is_owner_message(message):
            return
        await self._safe_answer(message, self.health_monitor.get_summary(), parse_mode=ParseMode.MARKDOWN)

    async def _cmd_backup(self, message: Message):
        if not self._is_owner_message(message):
            return
        try:
            self.backup._backup_now()
            backups = self.backup.get_backups()
            latest = backups[0].name if backups else "none"
            await self._safe_answer(
                message,
                f"✅ Backup selesai!\nFile terbaru: `{latest}`",
                parse_mode=ParseMode.MARKDOWN,
            )
        except Exception as e:
            logger.error("Manual backup failed: %s", e)
            await self._safe_answer(message, f"❌ Backup gagal: {str(e)[:200]}")

    async def _cmd_backups(self, message: Message):
        if not self._is_owner_message(message):
            return
        try:
            backups = self.backup.get_backups()
            if not backups:
                await self._safe_answer(message, "Belum ada backup.")
                return
            lines = ["📦 **Backup tersedia:**"]
            for i, path in enumerate(backups[:10], 1):
                size_mb = path.stat().st_size / (1024 * 1024)
                lines.append(f"{i}. `{path.name}` ({size_mb:.2f} MB)")
            await self._safe_answer(message, "\n".join(lines), parse_mode=ParseMode.MARKDOWN)
        except Exception as e:
            logger.error("List backups failed: %s", e)
            await self._safe_answer(message, f"❌ Gagal list backup: {str(e)[:200]}")

    async def _cmd_whoami(self, message: Message):
        user = message.from_user
        if not user or not self._is_message_authorized(message):
            return

        profile = self.learner.get_user_profile(user.id)
        name = profile.get("name")
        nickname = profile.get("nickname")
        job = profile.get("job")
        interests = profile.get("interests")
        preferences = profile.get("preferences")

        lines = ["👤 **Profil yang dikenali:**"]
        if name:
            lines.append(f"• Nama: {name}")
        if nickname:
            lines.append(f"• Panggilan: {nickname}")
        if job:
            lines.append(f"• Pekerjaan: {job}")
        if interests:
            lines.append(f"• Minat: {interests}")
        if preferences:
            lines.append(f"• Preferensi: {preferences}")
        if not any([name, nickname, job, interests, preferences]):
            lines.append("Belum ada data profil.")
            lines.append("Gunakan /intro untuk mengisi profilmu!")

        lines.append("")
        lines.append(self.learner.get_user_stats(user.id))
        await self._safe_answer(message, "\n".join(lines), parse_mode=ParseMode.MARKDOWN)

    async def _cmd_intro(self, message: Message):
        user = message.from_user
        if not user or not self._is_message_authorized(message):
            return

        args = self._command_args(message)
        if not args:
            await self._safe_answer(
                message,
                (
                    "💡 **Mau aku kenali kamu?**\n\n"
                    "Kamu bisa bilang langsung tanpa perintah:\n"
                    "_nama saya Budi, developer fullstack, suka Python & React_\n\n"
                    "Atau pakai format terstruktur:\n"
                    "`/intro Nama: Budi | Job: Developer | Minat: AI | Panggil: mas`\n\n"
                    "Aku bakal otomatis save dan ingat terus! 🧠"
                ),
                parse_mode=ParseMode.MARKDOWN,
            )
            return

        profile = self._parse_intro_args(args)
        stored = self.learner.save_user_profile(
            user_id=user.id,
            name=profile.get("nama", ""),
            nickname=profile.get("panggil", ""),
            job=profile.get("job", ""),
            interests=profile.get("minat", ""),
            preferences=profile.get("preferensi", ""),
        )

        if stored:
            lines = ["✅ Profil tersimpan! Aku bakal ingat:"]
            if profile.get("nama"):
                lines.append(f"• Nama: {profile['nama']}")
            if profile.get("panggil"):
                lines.append(f"• Panggilan: {profile['panggil']}")
            if profile.get("job"):
                lines.append(f"• Pekerjaan: {profile['job']}")
            if profile.get("minat"):
                lines.append(f"• Minat: {profile['minat']}")
            if profile.get("preferensi"):
                lines.append(f"• Preferensi: {profile['preferensi']}")
            lines.append("")
            lines.append("Nanti kalo mau update, kirim /intro lagi ya!")
        else:
            lines = ["⚠️ Gagal menyimpan profil. Coba lagi nanti."]

        await self._safe_answer(message, "\n".join(lines))

    async def _cmd_config(self, message: Message):
        user = message.from_user
        if not user or not self._is_owner(user.id):
            await self._safe_answer(message, "⛔ Hanya owner yang bisa akses config.")
            return

        args = self._command_args(message)
        if not args:
            lines = ["⚙️ **Konfigurasi:**\n"]
            lines.append(f"enable_streaming: {self.config.enable_streaming}")
            lines.append(f"telegram_stream_mode: {self.config.telegram_stream_mode}")
            lines.append(f"fast_routing: {self.config.fast_routing}")
            lines.append(f"background_learning: {self.config.background_learning}")
            lines.append(f"autonomous_learning: {self.config.autonomous_learning}")
            lines.append(f"auto_learn_from_files: {self.config.auto_learn_from_files}")
            lines.append(f"auto_memory_markdown: {self.config.auto_memory_markdown}")
            lines.append(f"log_level: {self.config.log_level}")
            lines.append(f"stream_edit_interval: {self.config.stream_edit_interval}s")
            lines.append(f"stream_min_chars: {self.config.stream_min_chars}")
            lines.append(f"file_context_max_chars: {self.config.file_context_max_chars}")
            lines.append(f"enable_self_improve: {self.config.enable_self_improve}")
            lines.append("")
            lines.append("Gunakan `/config set <key> <value>` untuk mengubah.")
            lines.append("Gunakan `/config reload` untuk reload .env.")
            await self._safe_answer(message, "\n".join(lines), parse_mode=ParseMode.MARKDOWN)
            return

        parts = args.split(maxsplit=1)
        subcommand = parts[0].lower()

        if subcommand == "reload":
            try:
                import importlib
                import kitsune.config as config_module
                importlib.reload(config_module)
                await self._safe_answer(message, "✅ Config module direload. Restart bot untuk efek penuh.")
            except Exception as e:
                await self._safe_answer(message, f"❌ Gagal reload: {e}")
            return

        if subcommand == "set" and len(parts) > 1:
            rest = parts[1]
            kv = rest.split(maxsplit=1)
            if len(kv) < 2:
                await self._safe_answer(message, "Format: `/config set <key> <value>`")
                return
            key, value = kv[0], kv[1]
            safe_bool_keys = {
                "enable_streaming", "fast_routing", "background_learning",
                "autonomous_learning", "auto_learn_from_files", "auto_memory_markdown",
                "enable_self_improve",
            }
            safe_str_keys = {
                "telegram_stream_mode", "log_level",
            }
            safe_float_keys = {
                "stream_edit_interval",
            }
            safe_int_keys = {
                "stream_min_chars", "file_context_max_chars",
            }

            try:
                if key in safe_bool_keys:
                    parsed = value.lower() in {"true", "1", "yes", "on"}
                    setattr(self.config, key, parsed)
                elif key in safe_str_keys:
                    setattr(self.config, key, value)
                elif key in safe_float_keys:
                    setattr(self.config, key, float(value))
                elif key in safe_int_keys:
                    setattr(self.config, key, int(value))
                else:
                    await self._safe_answer(
                        message,
                        f"Key `{key}` tidak bisa diedit via chat (bukan safe config)."
                    )
                    return
                await self._safe_answer(message, f"✅ `{key}` = `{value}`")
            except Exception as e:
                await self._safe_answer(message, f"❌ Gagal set config: {e}")
            return

        await self._safe_answer(
            message,
            "Subcommand tidak dikenal.\n"
            "`/config` — lihat config\n"
            "`/config set <key> <value>` — edit\n"
            "`/config reload` — reload .env",
            parse_mode=ParseMode.MARKDOWN,
        )

    async def _cmd_terminal(self, message: Message):
        user = message.from_user
        if not user or not self._is_owner(user.id):
            await self._safe_answer(message, "⛔ Hanya owner yang bisa akses terminal.")
            return

        args = self._command_args(message)
        if not args:
            await self._safe_answer(
                message,
                "🖥️ **Terminal**\n\n"
                "Format: `/terminal <command>`\n"
                "Contoh: `/terminal ls -la`\n\n"
                "⚠️ Hati-hati, ini mengeksekusi shell langsung!",
                parse_mode=ParseMode.MARKDOWN,
            )
            return

        if not self.local_tools:
            await self._safe_answer(message, "Tool lokal belum aktif. Set ENABLE_LOCAL_TOOLS=true di .env.")
            return

        result = self.local_tools.run(f"shell {args}")
        prefix = "✅" if result.ok else "❌"
        output = result.output[:3500] if result.output else "(no output)"
        await self._safe_answer(message, f"{prefix}\n```\n{output}\n```", parse_mode=ParseMode.MARKDOWN)

    async def _cmd_search(self, message: Message):
        user = message.from_user
        if not user or not self._is_message_authorized(message):
            await self._safe_answer(message, "⛔ Maaf, kamu tidak punya akses ke bot ini.")
            return

        query = self._command_args(message)
        if not query:
            await self._safe_answer(
                message,
                "🔍 **Web Search**\n\n"
                "Format: `/search <query>`\n"
                "Contoh:\n"
                "`/search harga bitcoin hari ini`\n"
                "`/search berita AI terbaru`\n\n"
                "Atau tanya langsung tanpa /search — aku bakal otomatis cari kalau perlu!",
                parse_mode=ParseMode.MARKDOWN,
            )
            return

        if not self.web_search:
            await self._safe_answer(
                message,
                "🔍 Web search belum aktif. Set ENABLE_WEB_SEARCH=true di .env untuk mengaktifkan."
            )
            return

        await self._safe_send_chat_action(message)
        try:
            results = self.web_search.search(query)
            if not results:
                await self._safe_answer(message, "🔍 Tidak ada hasil untuk pencarian tersebut.")
                return

            lines = [f"🔍 **Hasil pencarian:** `{query}`\n"]
            for idx, result in enumerate(results, 1):
                lines.append(f"{idx}. **{result.title}**")
                lines.append(f"   {result.snippet[:200]}" if result.snippet else "")
                lines.append(f"   [Buka link]({result.url})")
                lines.append("")

            await self._safe_answer(message, "\n".join(lines), parse_mode=ParseMode.MARKDOWN)
        except Exception as e:
            logger.error("Web search failed: %s", e)
            await self._safe_answer(message, f"❌ Gagal melakukan pencarian: {e}")

    async def _cmd_remind(self, message: Message):
        user = message.from_user
        if not user or not self._is_message_authorized(message):
            return
        args = self._command_args(message)
        if not args:
            await self._safe_answer(
                message,
                "Kirim seperti ini:\n/remind 10m minum obat\n/remind 1h30m meeting\n/remind 2026-04-30 09:00 deadline",
            )
            return

        parts = args.split(maxsplit=1)
        if len(parts) < 2:
            await self._safe_answer(
                message, "Format: /remind <waktu> <pesan>\nContoh: /remind 10m minum obat"
            )
            return

        time_str, reminder_text = parts
        due = ReminderSystem.parse_time(time_str)
        if not due:
            await self._safe_answer(
                message,
                "Format waktu tidak dikenal. Gunakan:\n"
                "• `10m` (10 menit)\n"
                "• `1h30m` (1 jam 30 menit)\n"
                "• `2d` (2 hari)\n"
                "• `2026-04-30 09:00` (waktu absolut)",
            )
            return

        reminder = self.reminders.add(user.id, message.chat.id, reminder_text, due)
        relative = self._format_relative_time(due)
        await self._safe_answer(
            message,
            f"✅ Oke, aku akan ingatkan kamu {relative}:\n_{reminder_text}_",
            parse_mode=ParseMode.MARKDOWN,
        )

    async def _cmd_reminders(self, message: Message):
        user = message.from_user
        if not user or not self._is_message_authorized(message):
            return
        items = self.reminders.list_for_user(user.id)
        if not items:
            await self._safe_answer(message, "Tidak ada pengingat aktif.")
            return
        lines = ["⏰ Pengingat aktif:"]
        for item in items:
            relative = self._format_relative_time(item.due_at)
            lines.append(f"• `{item.id}` — {relative}: {item.text[:60]}")
        await self._safe_answer(message, "\n".join(lines), parse_mode=ParseMode.MARKDOWN)

    async def _cmd_cancel_reminder(self, message: Message):
        user = message.from_user
        if not user or not self._is_message_authorized(message):
            return
        args = self._command_args(message)
        if not args:
            await self._safe_answer(message, "Kirim: /cancel_reminder <id>")
            return
        if self.reminders.cancel(user.id, args.strip()):
            await self._safe_answer(message, "✅ Pengingat dibatalkan.")
        else:
            await self._safe_answer(message, "Pengingat tidak ditemukan atau sudah terkirim.")

    # ---- File Generation ----

    @staticmethod
    def _detect_file_generation_request(text: str) -> str | None:
        """Detect if user is asking to generate/create a file. Returns suggested filename."""
        text_lower = text.lower()

        # Must have creation intent
        create_keywords = r"\b(buatkan|buat|generate|create|tulis|write|simpan|save|export|ekspor|bikin)\b"
        if not re.search(create_keywords, text_lower):
            return None

        # Must mention a file type
        file_indicators = r"\b(file|script|kode|code|program|skrip|konfigurasi|config|dokumen|document|\.py|\.js|\.json|\.yaml|\.yml|\.md|\.txt|\.sh|\.bat|\.html|\.css|\.sql|\.dockerfile)\b"
        if not re.search(file_indicators, text_lower):
            return None

        # Try to extract filename from the text
        filename_patterns = [
            r"\b(\w+[\w\-]*\.(?:py|js|json|yaml|yml|md|txt|sh|bat|html|css|sql|dockerfile|env|ini|conf|toml|go|rs|java|kt|swift|c|cpp|h))\b",
            r"\b(nama file|filename|save as|simpan sebagai)\s*[:=]?\s*['\"]?([^'\"\s]{1,60}\.\w{2,8})['\"]?",
        ]

        for pattern in filename_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                fname = match.group(1) if len(match.groups()) == 1 else match.group(2)
                if fname and "." in fname:
                    return fname.strip()

        # Default filename based on extension hint
        ext_map = {
            "python": "generated.py",
            "javascript": "generated.js",
            "js": "generated.js",
            "json": "data.json",
            "yaml": "config.yaml",
            "yml": "config.yml",
            "html": "index.html",
            "css": "style.css",
            "sql": "query.sql",
            "bash": "script.sh",
            "shell": "script.sh",
            "docker": "Dockerfile",
            "dockerfile": "Dockerfile",
            "markdown": "document.md",
            "config": "config.yaml",
            "text": "document.txt",
        }
        for keyword, default_name in ext_map.items():
            if keyword in text_lower:
                return default_name

        return "generated.py"

    async def _generate_and_send_file(
        self,
        message: Message,
        user_message: str,
        filename: str,
        model: str,
        fallback_model: str | list[str],
        memory_context: str,
        conversation_history: list[dict],
        user_name: str | None = None,
    ) -> bool:
        """Generate a file via LLM and send it as an attachment."""
        logger.info("📄 Generating file '%s' for user request", filename)

        generation_prompt = (
            f"Generate ONLY the file content for '{filename}' based on the user's request. "
            f"Do NOT include explanations, markdown formatting outside code blocks, or conversational text. "
            f"Output the raw file content directly.\n\n"
            f"User request: {user_message}"
        )

        try:
            response = await self.brain.think(
                user_message=generation_prompt,
                model=model,
                fallback_model=fallback_model,
                memory_context=memory_context,
                conversation_history=conversation_history,
                user_name=user_name,
                personality_context="",
            )
        except Exception as e:
            logger.error("File generation LLM call failed: %s", e)
            return False

        content = response.content

        # Strip markdown code block wrapper if present
        content = content.strip()
        if content.startswith("```"):
            lines = content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            elif lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines)

        content = content.strip()
        if not content:
            logger.warning("Generated file content is empty")
            return False

        # Save to generated files dir
        gen_dir = self.config.data_dir / "generated"
        gen_dir.mkdir(parents=True, exist_ok=True)
        safe_name = self._safe_filename(filename)
        file_path = gen_dir / safe_name

        # Handle duplicate names
        if file_path.exists():
            stem = file_path.stem or "generated"
            suffix = file_path.suffix
            for idx in range(1, 1000):
                candidate = gen_dir / f"{stem}_{idx}{suffix}"
                if not candidate.exists():
                    file_path = candidate
                    safe_name = candidate.name
                    break

        try:
            file_path.write_text(content, encoding="utf-8")
            logger.info("📄 Saved generated file: %s (%d bytes)", file_path, len(content))
        except OSError as e:
            logger.error("Failed to save generated file: %s", e)
            return False

        # Send as document
        try:
            await self._safe_send_chat_action(message, action=ChatAction.UPLOAD_DOCUMENT)
            input_file = FSInputFile(file_path)
            caption = f"📝 File '{safe_name}' berhasil dibuat!\n\n💡 Prompt: {user_message[:200]}"
            await message.answer_document(document=input_file, caption=caption)
            logger.info("📄 Sent generated file to chat: %s", safe_name)
            return True
        except Exception as e:
            logger.error("Failed to send generated file: %s", e)
            await self._safe_answer(message, f"❌ File dibuat tapi gagal dikirim: {e}")
            return False

    async def _handle_document_message(self, message: Message):
        user = message.from_user
        if not user or not self._is_message_authorized(message):
            await self._safe_answer(message, "⛔ Maaf, kamu tidak punya akses ke bot ini.")
            return
        if not message.document:
            return

        document = message.document
        filename = document.file_name or f"document_{document.file_unique_id}"
        await self._download_and_inspect_telegram_file(
            message=message,
            file_id=document.file_id,
            filename=filename,
            size_bytes=document.file_size or 0,
        )

    async def _handle_photo_message(self, message: Message):
        user = message.from_user
        if not user or not self._is_message_authorized(message):
            await self._safe_answer(message, "⛔ Maaf, kamu tidak punya akses ke bot ini.")
            return
        if not message.photo:
            return

        photo = message.photo[-1]
        await self._download_and_inspect_telegram_file(
            message=message,
            file_id=photo.file_id,
            filename=f"photo_{photo.file_unique_id}.jpg",
            size_bytes=photo.file_size or 0,
        )

    async def _handle_sticker(self, message: Message):
        user = message.from_user
        if not user or not self._is_message_authorized(message):
            return
        sticker = message.sticker
        if not sticker:
            return
        emoji = sticker.emoji or "🦊"
        await self._safe_answer(
            message,
            f"{emoji} Sticker lucu! Sayangnya aku belum bisa baca isi sticker, tapi aku siap bantu kalau ada yang lain."
        )

    async def _handle_unsupported_media(self, message: Message):
        user = message.from_user
        if not user or not self._is_message_authorized(message):
            return
        await self._safe_answer(
            message,
            "🎬 Media ini belum bisa aku proses. Coba kirim teks, foto, atau dokumen ya!"
        )

    async def _download_and_inspect_telegram_file(
        self,
        message: Message,
        file_id: str,
        filename: str,
        size_bytes: int,
    ):
        max_bytes = self.config.telegram_file_read_max_bytes
        if size_bytes and size_bytes > max_bytes:
            await self._safe_answer(
                message,
                f"File terlalu besar untuk dibaca: {format_size(size_bytes)}. "
                f"Limit saat ini {format_size(max_bytes)}.",
            )
            return

        await self._safe_send_chat_action(message)
        try:
            telegram_file = await message.bot.get_file(file_id)
        except (TelegramBadRequest, TelegramNetworkError, TelegramRetryAfter) as e:
            logger.warning("Telegram get_file failed: %s", e)
            await self._safe_answer(message, f"Gagal mengambil metadata file: {e}")
            return
        if not telegram_file.file_path:
            await self._safe_answer(message, "Telegram tidak mengembalikan path file untuk diunduh.")
            return

        destination = self._next_upload_path(message, filename)
        try:
            await message.bot.download_file(telegram_file.file_path, destination=destination)
        except (TelegramBadRequest, TelegramNetworkError, TelegramRetryAfter, OSError) as e:
            logger.warning("Telegram file download failed: %s", e)
            await self._safe_answer(message, f"Gagal download file: {e}")
            return

        try:
            inspection = inspect_file(destination, str(destination.relative_to(self.config.telegram_upload_dir.parent)))
        except OSError as e:
            await self._safe_answer(message, f"Gagal membaca file: {e}")
            return

        user = message.from_user
        if not user:
            await self._safe_answer(message, self._telegram_text_limit(inspection.render()))
            return

        caption = (message.caption or "").strip()
        await self.learner.learn_from_file(
            user_id=user.id,
            filename=filename,
            file_kind=inspection.kind,
            mime_type=inspection.mime_type,
            text_preview=inspection.preview,
            user_caption=caption,
        )

        if inspection.is_text and inspection.preview:
            await self._answer_from_uploaded_file(message, inspection, caption)
            return

        await self._safe_answer(message, self._telegram_text_limit(inspection.render()))

    def _get_user_name(self, user_id: int) -> str | None:
        identity = self.memory.get_user_identity(user_id)
        return identity.get("nickname") or identity.get("name")

    async def _answer_from_uploaded_file(self, message: Message, inspection, caption: str):
        user = message.from_user
        if not user:
            return

        interaction_id = f"int_{uuid.uuid4().hex[:12]}"
        hist_key = self._history_key(message)
        user_name = self._get_user_name(user.id)
        request = caption or "Baca file ini, jelaskan jenis file, isi pentingnya, dan hal yang perlu diperhatikan."
        file_context = inspection.preview[: self.config.file_context_max_chars]
        file_prompt = (
            f"User mengirim file Telegram.\n"
            f"Nama/path: {inspection.display_path}\n"
            f"Jenis: {inspection.kind}\n"
            f"MIME: {inspection.mime_type}\n"
            f"Ukuran: {format_size(inspection.size_bytes)}\n\n"
            f"Permintaan user: {request}\n\n"
            f"Isi file yang bisa dibaca:\n"
            f"```text\n{file_context}\n```"
        )

        try:
            await self._safe_send_chat_action(message)
            task_category, _ = await self.router.classify_task(file_prompt)
            primary_model, fallback_model = self.router.get_model_for_task(task_category)
            memory_context = "\n\n".join(
                part
                for part in [
                    self.learner.get_user_profile_context(user.id),
                    self.memory.get_user_context(user.id, request),
                ]
                if part
            )
            personality_context = self._get_personality_context(user.id)
            history = self._chat_history.get(hist_key, [])
            response = await self.brain.think(
                user_message=file_prompt,
                model=primary_model,
                fallback_model=fallback_model,
                memory_context=memory_context,
                conversation_history=history,
                user_name=user_name,
                personality_context=personality_context,
            )

            # AI autonomous shell execution (owner only)
            if self._is_owner(user.id) and "[[shell:" in response.content:
                response = await self._process_ai_shell_commands(
                    response=response,
                    user_message=file_prompt,
                    model=primary_model,
                    fallback_model=fallback_model,
                    memory_context=memory_context,
                    conversation_history=history,
                    user_name=user_name,
                    personality_context=personality_context,
                )

            self._chat_history.setdefault(hist_key, [])
            self._chat_history[hist_key].append({"role": "user", "content": f"[file] {inspection.display_path}\n{request}"})
            self._chat_history[hist_key].append({"role": "assistant", "content": response.content})
            if len(self._chat_history[hist_key]) > 20:
                self._chat_history[hist_key] = self._chat_history[hist_key][-20:]

            await self._send_or_finalize_response(message, response, task_category, interaction_id)

            # Store response metadata for implicit feedback detection
            self._store_response_meta(
                user_id=user.id,
                model_used=response.model_used,
                task_category=task_category,
                response_text=response.content,
                response_time=response.response_time,
            )

            learning_args = dict(
                user_id=user.id,
                user_message=f"[uploaded_file] {inspection.display_path}\n{request}\n{file_context[:1200]}",
                bot_response=response.content,
                task_category=task_category,
                model_used=response.model_used,
                response_time=response.response_time,
                response_success=self.learner.assess_response_success(response.content, response.success),
                interaction_id=interaction_id,
                error_type=response.error_type,
            )
            if self.config.background_learning:
                asyncio.create_task(self._learn_after_response(**learning_args))
            else:
                await self._learn_after_response(**learning_args)
        except Exception as e:
            logger.error("Error answering uploaded file: %s\n%s", e, traceback.format_exc())
            await self._safe_answer(
                message,
                self._telegram_text_limit(
                    inspection.render()
                    + "\n\nFile terbaca, tapi gagal dianalisis oleh model. "
                    + f"Error: {str(e)[:180]}"
                ),
            )

    async def _handle_message(self, message: Message):
        user = message.from_user

        if not user or not self._is_message_authorized(message):
            await self._safe_answer(message, "⛔ Maaf, kamu tidak punya akses ke bot ini.")
            return

        text = message.text or ""
        if not text.strip():
            return

        # Check implicit negative feedback from previous interaction
        if await self._check_implicit_signals(user.id, text):
            # Still continue processing the new message, but feedback was recorded
            pass

        # Store current message for future similarity checks
        self._last_user_message[user.id] = text

        interaction_id = f"int_{uuid.uuid4().hex[:12]}"
        hist_key = self._history_key(message)
        user_name = self._get_user_name(user.id)
        logger.info("📩 Message from %s (%d): %s", user.first_name, user.id, text[:100])

        try:
            await self._safe_send_chat_action(message)

            task_category, _ = await self.router.classify_task(text)
            primary_model, fallback_model = self.router.get_model_for_task(task_category)

            # Check for model override request
            requested_override = self._detect_model_switch_request(text)
            if requested_override:
                self._user_model_override[user.id] = requested_override
                provider = requested_override.split("/")[0] if "/" in requested_override else "unknown"
                await self._safe_answer(
                    message,
                    f"✅ Oke, sekarang aku pakai **{requested_override.split('/')[-1]}** ({provider}) "
                    f"untuk semua pesanmu.\n\n"
                    f"Ketik *'batal model'* atau *'reset model'* untuk kembali ke auto-routing.",
                    parse_mode=ParseMode.MARKDOWN,
                )
                logger.info("🔄 User %d set model override: %s", user.id, requested_override)
                return

            if self._parse_reset_model(text):
                old_override = self._user_model_override.pop(user.id, None)
                if old_override:
                    await self._safe_answer(
                        message,
                        f"🔄 Model override dibatalkan. Sekarang aku balik ke auto-routing! 🦊",
                    )
                    logger.info("🔄 User %d reset model override", user.id)
                else:
                    await self._safe_answer(message, "Tidak ada model override yang aktif.")
                return

            # Apply active model override if present
            override = self._user_model_override.get(user.id)
            if override:
                primary_model = override
                fallback_model = []  # No fallback when user explicitly chose
                logger.info("🔄 Using user-override model %s for user %d", override, user.id)

            # Check for conversational personality hints
            requested_personality = self._detect_personality_change(text, user.id)
            if requested_personality:
                self.memory.set_user_personality(user.id, requested_personality)
                # Respond naturally as if discussing, not as a system confirmation
                await self._safe_answer(message, "Oke, noted! 📝")
                logger.info("🎭 User %d personality auto-detected: %s", user.id, requested_personality[:100])
                # Continue processing the original message, not return

            if self._detect_personality_reset(text):
                old = self.memory.get_user_personality(user.id)
                if old:
                    self.memory.set_user_personality(user.id, "")
                    await self._safe_answer(message, "Oke, balik normal lagi ya! 🦊")
                    logger.info("🎭 User %d reset personality", user.id)
                else:
                    await self._safe_answer(message, "Belum ada gaya khusus yang aku pake sih.")
                return

            # Build personality context for this user
            personality_context = self._get_personality_context(user.id)

            # Auto file generation
            suggested_filename = self._detect_file_generation_request(text)
            if suggested_filename:
                await self._safe_send_chat_action(message, action=ChatAction.UPLOAD_DOCUMENT)
                memory_context = "\n\n".join(
                    part
                    for part in [
                        self.learner.get_user_profile_context(user.id),
                        self.memory.get_user_context(user.id, text),
                    ]
                    if part
                )
                history = self._chat_history.get(hist_key, [])
                generated = await self._generate_and_send_file(
                    message=message,
                    user_message=text,
                    filename=suggested_filename,
                    model=primary_model,
                    fallback_model=fallback_model,
                    memory_context=memory_context,
                    conversation_history=history,
                    user_name=user_name,
                )
                if generated:
                    return
                # If generation failed, fall through to normal response

            # Auto web search for real-time queries
            search_context = ""
            if self.web_search and task_category == "web_search":
                try:
                    results = self.web_search.search(text)
                    if results:
                        search_context = self.web_search.format_for_prompt(results)
                        logger.info("🔍 Auto-search injected %d results for web_search task", len(results))
                except Exception as e:
                    logger.warning("Auto web search failed: %s", e)

            memory_context = "\n\n".join(
                part
                for part in [
                    self.learner.get_user_profile_context(user.id),
                    self.memory.get_user_context(user.id, text),
                    search_context,
                ]
                if part
            )
            history = self._chat_history.get(hist_key, [])

            if self.config.enable_streaming:
                response = await self._stream_response(
                    message=message,
                    user_message=text,
                    model=primary_model,
                    fallback_model=fallback_model,
                    memory_context=memory_context,
                    conversation_history=history,
                    user_name=user_name,
                    personality_context=personality_context,
                )
            else:
                response = await self.brain.think(
                    user_message=text,
                    model=primary_model,
                    fallback_model=fallback_model,
                    memory_context=memory_context,
                    conversation_history=history,
                    user_name=user_name,
                    personality_context=personality_context,
                )

            # AI autonomous shell execution (owner only)
            if self._is_owner(user.id) and "[[shell:" in response.content:
                response = await self._process_ai_shell_commands(
                    response=response,
                    user_message=text,
                    model=primary_model,
                    fallback_model=fallback_model,
                    memory_context=memory_context,
                    conversation_history=history,
                    user_name=user_name,
                    personality_context=personality_context,
                )

            self._chat_history.setdefault(hist_key, [])
            self._chat_history[hist_key].append({"role": "user", "content": text})
            self._chat_history[hist_key].append({"role": "assistant", "content": response.content})
            if len(self._chat_history[hist_key]) > 20:
                self._chat_history[hist_key] = self._chat_history[hist_key][-20:]

            await self._send_or_finalize_response(message, response, task_category, interaction_id)

            # Store response metadata for implicit feedback detection
            self._store_response_meta(
                user_id=user.id,
                model_used=response.model_used,
                task_category=task_category,
                response_text=response.content,
                response_time=response.response_time,
            )

            # Detect if user shared profile info and acknowledge
            await self._maybe_acknowledge_learned_profile(message, text)

            learning_args = dict(
                user_id=user.id,
                user_message=text,
                bot_response=response.content,
                task_category=task_category,
                model_used=response.model_used,
                response_time=response.response_time,
                response_success=self.learner.assess_response_success(response.content, response.success),
                interaction_id=interaction_id,
                error_type=response.error_type,
            )
            if self.config.background_learning:
                asyncio.create_task(self._learn_after_response(**learning_args))
            else:
                await self._learn_after_response(**learning_args)

        except Exception as e:
            await self._handle_message_error(message, text, user, hist_key, e)

    async def _handle_group_mention(self, message: Message):
        """Handle @bot mentions and replies to bot in groups."""
        user = message.from_user
        if not user or not self._is_message_authorized(message):
            return

        # Must be a group
        if message.chat.type not in {ChatType.GROUP, ChatType.SUPERGROUP}:
            return

        text = message.text or ""
        if not text.strip():
            return

        is_reply_to_bot = False
        is_mention = False
        extracted_text = text.strip()

        # Check if this is a reply to bot's message
        if message.reply_to_message and self._bot_id is not None:
            reply_from = message.reply_to_message.from_user
            if reply_from and reply_from.id == self._bot_id:
                is_reply_to_bot = True

        # Check if bot is mentioned (@username)
        if self._bot_username and f"@{self._bot_username}" in text:
            is_mention = True
            # Remove mention from text, handling surrounding whitespace/punctuation
            extracted_text = re.sub(
                rf"(?:^|\s)@{re.escape(self._bot_username)}\b[\s:;,.​]*",
                " ",
                text,
            ).strip()

        if not is_reply_to_bot and not is_mention:
            return

        # Skip if extracted text is empty after removing mention
        if not extracted_text:
            extracted_text = "..."

        logger.info("📩 Group mention/reply from %s (%d): %s", user.first_name, user.id, extracted_text[:100])

        interaction_id = f"int_{uuid.uuid4().hex[:12]}"
        hist_key = self._history_key(message)
        user_name = self._get_user_name(user.id)

        try:
            await self._safe_send_chat_action(message)

            # Check implicit signals
            if await self._check_implicit_signals(user.id, extracted_text):
                pass
            self._last_user_message[user.id] = extracted_text

            task_category, _ = await self.router.classify_task(extracted_text)
            primary_model, fallback_model = self.router.get_model_for_task(task_category)

            # Apply model override if present
            override = self._user_model_override.get(user.id)
            if override:
                primary_model = override
                fallback_model = []
                logger.info("🔄 Using user-override model %s for user %d in group", override, user.id)

            # Check for conversational personality hints in group
            requested_personality = self._detect_personality_change(extracted_text, user.id)
            if requested_personality:
                self.memory.set_user_personality(user.id, requested_personality)
                logger.info("🎭 User %d personality auto-detected in group: %s", user.id, requested_personality[:100])

            personality_context = self._get_personality_context(user.id)
            memory_context = "\n\n".join(
                part
                for part in [
                    self.learner.get_user_profile_context(user.id),
                    self.memory.get_user_context(user.id, extracted_text),
                ]
                if part
            )
            history = self._chat_history.get(hist_key, [])

            if self.config.enable_streaming:
                response = await self._stream_response(
                    message=message,
                    user_message=extracted_text,
                    model=primary_model,
                    fallback_model=fallback_model,
                    memory_context=memory_context,
                    conversation_history=history,
                    user_name=user_name,
                    personality_context=personality_context,
                )
            else:
                response = await self.brain.think(
                    user_message=extracted_text,
                    model=primary_model,
                    fallback_model=fallback_model,
                    memory_context=memory_context,
                    conversation_history=history,
                    user_name=user_name,
                    personality_context=personality_context,
                )

            # AI autonomous shell execution (owner only)
            if self._is_owner(user.id) and "[[shell:" in response.content:
                response = await self._process_ai_shell_commands(
                    response=response,
                    user_message=extracted_text,
                    model=primary_model,
                    fallback_model=fallback_model,
                    memory_context=memory_context,
                    conversation_history=history,
                    user_name=user_name,
                    personality_context=personality_context,
                )

            self._chat_history.setdefault(hist_key, [])
            self._chat_history[hist_key].append({"role": "user", "content": extracted_text})
            self._chat_history[hist_key].append({"role": "assistant", "content": response.content})
            if len(self._chat_history[hist_key]) > 20:
                self._chat_history[hist_key] = self._chat_history[hist_key][-20:]

            # Send as threaded reply
            await self._send_or_finalize_response_group(message, response, task_category, interaction_id)

            self._store_response_meta(
                user_id=user.id,
                model_used=response.model_used,
                task_category=task_category,
                response_text=response.content,
                response_time=response.response_time,
            )

            learning_args = dict(
                user_id=user.id,
                user_message=extracted_text,
                bot_response=response.content,
                task_category=task_category,
                model_used=response.model_used,
                response_time=response.response_time,
                response_success=self.learner.assess_response_success(response.content, response.success),
                interaction_id=interaction_id,
                error_type=response.error_type,
            )
            if self.config.background_learning:
                asyncio.create_task(self._learn_after_response(**learning_args))
            else:
                await self._learn_after_response(**learning_args)

        except Exception as e:
            await self._handle_message_error(message, extracted_text, user, hist_key, e)

    async def _send_or_finalize_response_group(
        self,
        message: Message,
        response: BrainResponse,
        task_category: str,
        interaction_id: str | None,
    ):
        """Same as _send_or_finalize_response but sends as a threaded reply."""
        model_short = response.model_used.split("/")[-1] if "/" in response.model_used else response.model_used
        footer = f"\n\n`🤖 {model_short} | 📋 {task_category} | ⏱ {response.response_time}s`"
        keyboard = self._build_feedback_keyboard()

        stream_message = getattr(response, "_telegram_message", None)
        is_draft = getattr(response, "_is_draft", False)
        chunks = self._split_text(response.content, 4096)

        reply_kwargs = {"reply_to_message_id": message.message_id}

        if is_draft:
            stream_message = None

        if len(chunks) == 1 and len(chunks[0]) + len(footer) <= 4096:
            reply_text = chunks[0] + footer
            if stream_message:
                edited = False
                try:
                    await stream_message.edit_text(
                        text=reply_text,
                        parse_mode=ParseMode.MARKDOWN,
                        reply_markup=keyboard,
                    )
                    edited = True
                except TelegramBadRequest:
                    edited = await self._safe_edit_text(stream_message, reply_text, reply_markup=keyboard)
                if edited:
                    self._try_register_feedback_meta(stream_message, message, response, task_category, interaction_id)
                    return
                try:
                    await stream_message.delete()
                except Exception:
                    pass

            sent = await self._safe_answer(message, reply_text, parse_mode=ParseMode.MARKDOWN, reply_markup=keyboard, **reply_kwargs)
            self._try_register_feedback_meta(sent, message, response, task_category, interaction_id)
            return

        # Multi-chunk: delete stream message and send all chunks fresh
        if stream_message and len(chunks) > 1:
            try:
                await stream_message.delete()
            except Exception:
                pass
            stream_message = None

        for idx, chunk in enumerate(chunks):
            is_last = idx == len(chunks) - 1
            text = chunk + footer if is_last else chunk
            markup = keyboard if is_last else None
            kwargs = {"parse_mode": ParseMode.MARKDOWN if not is_last else None, "reply_markup": markup, **reply_kwargs}
            sent = await self._safe_answer(message, text, **kwargs)
            if is_last:
                self._try_register_feedback_meta(sent, message, response, task_category, interaction_id)

    async def _on_reminder_trigger(self, reminder):
        if not self.bot:
            return
        try:
            await self.bot.send_message(
                chat_id=reminder.chat_id,
                text=f"⏰ Pengingat:\n\n{reminder.text}",
            )
        except Exception as e:
            logger.warning("Failed to send reminder: %s", e)

    async def _error_handler(self, event):
        exc = event.exception
        logger.error("Unhandled aiogram exception: %s", exc, exc_info=exc)

        # Auto-generate improvement proposal for unhandled exceptions
        if self.self_improver:
            try:
                self.self_improver.record_negative_feedback(
                    user_id=0,
                    user_message="[aiogram unhandled error]",
                    bot_response=str(exc)[:500],
                    task_category="system",
                    model_used="none",
                )
            except Exception:
                pass

    async def _handle_message_error(
        self,
        message: Message,
        user_message: str,
        user,
        hist_key: tuple[int, int],
        error: Exception,
    ):
        """Self-healing error recovery for message handling failures."""
        error_msg = str(error)
        error_lower = error_msg.lower()
        exc_type = type(error).__name__

        logger.error("Error handling message: %s\n%s", error, traceback.format_exc())

        # Classify error
        is_llm_error = any(k in error_lower for k in (
            "timeout", "rate limit", "ratelimit", "context length",
            "token limit", "authentication", "unauthorized", "401", "403",
            "litellm", "openai", "anthropic", "gemini", "openrouter",
        ))
        is_telegram_error = "telegram" in error_lower or "bad request" in error_lower
        is_too_long = "message is too long" in error_lower

        # Strategy 1: Telegram message too long — ultimate fallback truncation
        if is_too_long:
            try:
                truncated = user_message[:3950]
                await self._safe_answer(
                    message,
                    truncated + "\n\n...[respons terlalu panjang, dipotong]",
                )
                return
            except Exception:
                pass

        # Strategy 2: LLM error — auto-retry once with fastest available model
        if is_llm_error and not is_telegram_error:
            try:
                logger.info("🔄 Auto-retrying with fast model due to: %s", exc_type)
                fast_model = pick_fast_model(self.config)
                history = self._chat_history.get(hist_key, [])
                personality_context = self._get_personality_context(user.id) if user else ""
                retry_response = await self.brain.think(
                    user_message=user_message,
                    model=fast_model,
                    fallback_model=[],
                    memory_context="",
                    conversation_history=history,
                    user_name=user.first_name if user else None,
                    personality_context=personality_context,
                )
                if retry_response.success:
                    await self._send_or_finalize_response(message, retry_response, "simple_qa", None)
                    self._chat_history.setdefault(hist_key, [])
                    self._chat_history[hist_key].append({"role": "user", "content": user_message})
                    self._chat_history[hist_key].append({"role": "assistant", "content": retry_response.content})
                    if len(self._chat_history[hist_key]) > 20:
                        self._chat_history[hist_key] = self._chat_history[hist_key][-20:]
                    logger.info("✅ Auto-retry with %s succeeded", fast_model)
                    return
            except Exception as retry_err:
                logger.warning("Auto-retry failed: %s", retry_err)

        # Strategy 3: Record error for learning and routing optimization
        try:
            error_type = self.brain._classify_error(error)
        except Exception:
            error_type = "unknown"
        try:
            self.router.record_result(
                task_category="unknown",
                model_used="none",
                success=False,
                response_time=0.0,
                error_type=error_type,
            )
        except Exception:
            pass

        # Strategy 4: Auto-generate improvement proposal
        if self.self_improver:
            try:
                self.self_improver.record_negative_feedback(
                    user_id=user.id if user else 0,
                    user_message=user_message,
                    bot_response=f"[ERROR] {exc_type}: {error_msg[:400]}",
                    task_category="unknown",
                    model_used="none",
                )
                logger.info("📝 Auto-generated improvement proposal from error recovery")
            except Exception:
                pass

        # Strategy 5: Friendly user message (no raw error exposed)
        friendly = self._friendly_error_message(error)
        await self._safe_answer(message, friendly)

    @staticmethod
    def _friendly_error_message(error: Exception) -> str:
        """Return a user-friendly message, hiding technical details."""
        msg = str(error).lower()
        if any(k in msg for k in ("message is too long", "too long")):
            return "😅 Responsnya terlalu panjang untuk Telegram. Coba tanyakan yang lebih spesifik ya!"
        if any(k in msg for k in ("timeout", "timed out", "connection")):
            return "⏱️ Koneksi ke model AI sedang lambat. Coba lagi dalam beberapa detik ya!"
        if any(k in msg for k in ("rate limit", "ratelimit", "too many requests")):
            return "🚦 Terlalu banyak permintaan ke AI. Tunggu sebentar dan coba lagi ya!"
        if any(k in msg for k in ("authentication", "unauthorized", "api key", "invalid key")):
            return "🔐 Ada masalah autentikasi dengan provider AI. Hubungi owner ya!"
        if any(k in msg for k in ("context length", "token limit", "too large")):
            return "📏 Percakapan terlalu panjang. Coba /reset untuk mulai dari awal!"
        if "telegram" in msg:
            return "📡 Ada masalah dengan Telegram. Coba lagi ya!"
        return "😅 Maaf, saya sedang mengalami masalah teknis. Coba lagi nanti ya!"

    # ---- Personality ----

    def _get_personality_context(self, user_id: int) -> str:
        """Return the user's preferred personality text for injection into system prompt."""
        personality = self.memory.get_user_personality(user_id)
        return personality

    def _detect_personality_change(self, text: str, user_id: int) -> str | None:
        """Detect conversational personality hints from normal chat."""
        text_lower = text.lower().strip()

        # Skip if user is just asking about the bot's nature
        if any(k in text_lower for k in ("siapa kamu", "kamu siapa", "who are you", "apa itu")):
            return None

        # Pattern: direct behavioral feedback / request
        feedback_patterns = [
            r"\b(?:jadi|buat)\s+(?:kamu|lu|kau|elmu|elo|loe|bot)\s+(?:jadi|menjadi|ber|lebih|agak|sedikit)\s+(.{5,200})\b",
            r"\b(?:kamu|lu|kau|elmu|elo|loe|bot)\s+(?:harus|mesti|sebaiknya|coba|tolong)\s+(?:jadi|menjadi|ber|lebih|agak|sedikit)\s+(.{5,200})\b",
            r"\b(?:kamu|lu|kau|elmu|elo|loe|bot)\s+(?:terlalu|kelewatan|kurang)\s+(.{3,120})\b",
            r"\b(?:kamu|lu|kau|elmu|elo|loe|bot)\s+(?:itu|nya|ini)\s+(?:terlalu|kelewatan|kurang)\s+(.{3,120})\b",
            r"\b(?:lebih|jadi)\s+(.{5,200})\s+(?:dong|deh|ya|sih|lah)\b",
            r"\b(?:santai|formal|serius|lucu|humor|sarkas|galak|lembut|ramah|dingin|hangat|cepat|lambat|pendek|panjang|detail|singkat)\s+(?:dong|deh|ya|sih|lah|aja|saja)\b",
        ]

        for pattern in feedback_patterns:
            match = re.search(pattern, text_lower)
            if match:
                trait = match.group(1).strip(" .,;!?")
                # Build a full personality description from the trait
                return self._build_personality_from_trait(trait, user_id)

        # Pattern: single-word/adjective personality hints
        single_trait_patterns = [
            r"\b(?:kamu|lu|kau|elmu|elo|loe|bot)\s+(?:jadi|menjadi|ber|lebih)\s+(.{3,80})\b",
        ]
        for pattern in single_trait_patterns:
            match = re.search(pattern, text_lower)
            if match:
                trait = match.group(1).strip(" .,;!?")
                return self._build_personality_from_trait(trait, user_id)

        return None

    @staticmethod
    def _build_personality_from_trait(trait: str, user_id: int) -> str:
        """Expand a short trait into a fuller personality description."""
        trait_lower = trait.lower()

        # Known trait expansions
        expansions = {
            "santai": "Berperilaku santai, casual, dan tidak kaku. Gunakan bahasa sehari-hari.",
            "formal": "Berperilaku formal, sopan, dan profesional. Gunakan bahasa baku.",
            "serius": "Berperilaku serius dan fokus pada fakta. Hindari lelucon.",
            "lucu": "Berperilaku lucu dan humoris. Boleh bercanda sesekali.",
            "humor": "Berperilaku lucu dan humoris. Boleh bercanda sesekali.",
            "humoris": "Berperilaku lucu dan humoris. Boleh bercanda sesekali.",
            "sarkas": "Berperilaku sarkastik tapi tetap membantu. Gunakan sindiran halus.",
            "sarkastik": "Berperilaku sarkastik tapi tetap membantu. Gunakan sindiran halus.",
            "galak": "Berperilaku tegas dan galak, tapi tetap membantu.",
            "lembut": "Berperilaku lembut, sabar, dan penuh perhatian.",
            "ramah": "Berperilaku ramah, hangat, dan menyambut.",
            "dingin": "Berperilaku tenang, objektif, dan tidak terlalu ekspresif.",
            "hangat": "Berperilaku hangat, ramah, dan penuh empati.",
            "cepat": "Berikan jawaban singkat, padat, dan langsung ke inti.",
            "lambat": "Berikan jawaban detail dan teliti, tidak terburu-buru.",
            "pendek": "Jawaban singkat dan padat, maksimal 2-3 kalimat.",
            "panjang": "Jawaban detail dan komprehensif, jelaskan secara menyeluruh.",
            "detail": "Berikan jawaban yang sangat detail dengan contoh dan penjelasan mendalam.",
            "singkat": "Jawaban singkat dan padat, maksimal 2-3 kalimat.",
        }

        for key, expansion in expansions.items():
            if key in trait_lower:
                return expansion

        # Fallback: build a generic description
        return f"Berperilaku {trait}. Sesuaikan gaya komunikasi dengan deskripsi ini."

    @staticmethod
    def _detect_personality_reset(text: str) -> bool:
        """Detect personality reset requests in conversation."""
        text_lower = text.lower().strip()
        return bool(re.search(
            r"\b(?:reset|hapus|batal|kembali|default|normal|awal)\s+(?:personality|persona|karakter|gaya|sifat|tingkah)\b|"
            r"\b(?:balik|kembali)\s+ke\s+(?:default|awal|normal|semula)\b|"
            r"\b(?:jadi)\s+(?:normal|default|awal|semula)\s+(?:lagi|aja|saja)\b",
            text_lower,
        ))

    # ---- Model Override Detection ----

    @staticmethod
    def _detect_model_switch_request(text: str) -> str | None:
        """Detect if user wants to switch to a specific model. Returns canonical model or None."""
        text_lower = text.lower().strip()
        # Patterns like "gunakan kimi", "pakai codex", "ganti ke deepseek", etc.
        patterns = [
            r"\b(gunakan|pakai|ganti\s+ke|switch\s+to|use|pake)\s+([a-z0-9\-:]+)\b",
            r"\b(ganti\s+model\s+(?:jadi|ke|menjadi))\s+([a-z0-9\-:]+)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                alias = match.group(2).strip()
                resolved = resolve_model_alias(alias)
                if resolved:
                    return resolved
        return None

    @staticmethod
    def _parse_reset_model(text: str) -> bool:
        """Detect if user wants to reset/clear model override."""
        text_lower = text.lower().strip()
        return bool(re.search(
            r"\b(reset|batal|hapus|clear)\s+(?:model|override|pengaturan)\b|"
            r"\b(kembali|default)\s+(?:model|pengaturan)\b|"
            r"\b(balik)\s+ke\s+(?:auto|otomatis|default)\b",
            text_lower,
        ))

    # ---- Implicit Negative Feedback ----

    def _store_response_meta(self, user_id: int, model_used: str, task_category: str, response_text: str, response_time: float):
        """Store metadata of the last response for implicit feedback detection."""
        self._last_response_meta[user_id] = {
            "model_used": model_used,
            "task_category": task_category,
            "response_text": response_text,
            "timestamp": time.time(),
            "response_time": response_time,
        }

    async def _check_implicit_signals(self, user_id: int, user_message: str) -> bool:
        """
        Detect implicit negative signals from user message.
        Returns True if negative feedback was recorded.
        """
        text_lower = user_message.lower().strip()
        if len(text_lower) < 2:
            return False

        meta = self._last_response_meta.get(user_id)
        if not meta:
            return False

        signals: list[str] = []

        # Signal 1: Repeated message (user sends same/similar message again within 60s)
        last_msg = self._last_user_message.get(user_id, "")
        if last_msg and self._text_similarity(last_msg, user_message) > 0.75:
            time_since = time.time() - meta.get("timestamp", 0)
            if time_since < 60:
                signals.append(f"repeated_message ({time_since:.0f}s)")

        # Signal 2: Negative keywords
        for pattern in self._implicit_negative_keywords:
            if re.search(pattern, text_lower):
                signals.append("negative_keyword")
                break

        # Signal 3: Very slow response (> 60s) + short follow-up
        if meta.get("response_time", 0) > 60 and len(user_message) < 30:
            signals.append("slow_response_short_followup")

        # Signal 4: Very short or substance-less bot response
        response_text = meta.get("response_text", "")
        if len(response_text.strip()) < 30:
            signals.append("very_short_response")

        # Signal 5: User asks for retry/repeat explicitly
        if re.search(r"\b(ulangi|coba lagi|repeat|try again|sekali lagi)\b", text_lower):
            signals.append("explicit_retry_request")

        if not signals:
            return False

        logger.info(
            "🟡 Implicit negative signals detected for user %d: %s",
            user_id,
            ", ".join(signals),
        )

        # Record negative feedback through learner and self-improver
        try:
            await self.learner.process_feedback(
                user_id=user_id,
                interaction_id=None,
                feedback="negative",
                task_category=meta["task_category"],
                model_used=meta["model_used"],
            )
        except Exception as e:
            logger.debug("Implicit feedback learner recording failed: %s", e)

        if self.self_improver:
            try:
                self.self_improver.record_negative_feedback(
                    user_id=user_id,
                    user_message=self._last_user_message.get(user_id, ""),
                    bot_response=response_text,
                    task_category=meta["task_category"],
                    model_used=meta["model_used"],
                )
                logger.info("📝 Auto-generated improvement proposal from implicit signals")
            except Exception as e:
                logger.debug("Implicit feedback self-improver recording failed: %s", e)

        return True

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        """Simple Jaccard similarity for repeated message detection."""
        set_a = set(a.lower().split())
        set_b = set(b.lower().split())
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union else 0.0

    async def _learn_after_response(
        self,
        user_id: int,
        user_message: str,
        bot_response: str,
        task_category: str,
        model_used: str,
        response_time: float,
        response_success: bool = True,
        interaction_id: str | None = None,
        error_type: str | None = None,
    ):
        try:
            await self.learner.learn_from_interaction(
                user_id=user_id,
                user_message=user_message,
                bot_response=bot_response,
                task_category=task_category,
                model_used=model_used,
                response_time=response_time,
                response_success=response_success,
                interaction_id=interaction_id,
                error_type=error_type,
            )
        except Exception as e:
            logger.warning("Background learning failed: %s", e, exc_info=True)

    async def _stream_response(
        self,
        message: Message,
        user_message: str,
        model: str,
        fallback_model: str | list[str],
        memory_context: str,
        conversation_history: list[dict],
        user_name: str | None = None,
        personality_context: str = "",
    ) -> BrainResponse:
        if self._can_stream_with_draft(message):
            response = await self._stream_response_with_draft(
                message=message,
                user_message=user_message,
                model=model,
                fallback_model=fallback_model,
                memory_context=memory_context,
                conversation_history=conversation_history,
                user_name=user_name,
                personality_context=personality_context,
            )
            if response:
                return response

        stream_message = await self._safe_answer(message, "...")
        if not stream_message:
            return await self.brain.think(user_message, model, fallback_model, memory_context, conversation_history, user_name, personality_context)

        final_response = None
        content = ""
        last_edit_at = 0.0
        last_edit_len = 0

        async for event in self.brain.stream_think(user_message, model, fallback_model, memory_context, conversation_history, user_name, personality_context):
            if event.get("type") == "delta":
                content += event.get("content", "")
                now = time.monotonic()
                if (
                    now - last_edit_at >= self.config.stream_edit_interval
                    and len(content) - last_edit_len >= self.config.stream_min_chars
                ):
                    preview = self._stream_preview(content)
                    # Try with markdown first, fallback to plain text if invalid
                    ok = await self._safe_edit_text(stream_message, preview, parse_mode=ParseMode.MARKDOWN)
                    if not ok:
                        ok = await self._safe_edit_text(stream_message, preview)
                    if ok:
                        last_edit_at = now
                        last_edit_len = len(content)
            elif event.get("type") == "final":
                final_response = event["response"]

        if final_response is None:
            return await self.brain.think(user_message, model, fallback_model, memory_context, conversation_history, user_name)

        final_response._telegram_message = stream_message
        return final_response

    def _can_stream_with_draft(self, message: Message) -> bool:
        return self.config.telegram_stream_mode == "draft" and message.chat.type == ChatType.PRIVATE

    async def _stream_response_with_draft(
        self,
        message: Message,
        user_message: str,
        model: str,
        fallback_model: str | list[str],
        memory_context: str,
        conversation_history: list[dict],
        user_name: str | None = None,
        personality_context: str = "",
    ) -> BrainResponse | None:
        chat_id = message.chat.id
        draft_id = int(time.time() * 1000)
        final_response = None
        content = ""
        last_draft_at = 0.0
        last_draft_len = 0

        await self._safe_send_chat_action(message)

        async for event in self.brain.stream_think(user_message, model, fallback_model, memory_context, conversation_history, user_name, personality_context):
            if event.get("type") == "delta":
                content += event.get("content", "")
                now = time.monotonic()
                if (
                    now - last_draft_at >= self.config.stream_edit_interval
                    and len(content) - last_draft_len >= self.config.stream_min_chars
                ):
                    draft_status = await self._safe_send_message_draft(chat_id, draft_id, self._draft_preview(content))
                    if draft_status is True:
                        last_draft_at = now
                        last_draft_len = len(content)
                    elif draft_status is False:
                        return None
            elif event.get("type") == "final":
                final_response = event["response"]

        if final_response is None:
            return None

        # Clear the draft by sending empty draft
        try:
            await self._safe_send_message_draft(chat_id, draft_id, "")
        except Exception:
            pass

        # Mark as draft so _send_or_finalize_response knows to send fresh
        final_response._is_draft = True  # type: ignore[attr-defined]
        return final_response

    async def _send_or_finalize_response(
        self,
        message: Message,
        response: BrainResponse,
        task_category: str,
        interaction_id: str | None,
    ):
        model_short = response.model_used.split("/")[-1] if "/" in response.model_used else response.model_used
        footer = f"\n\n`🤖 {model_short} | 📋 {task_category} | ⏱ {response.response_time}s`"
        keyboard = self._build_feedback_keyboard()

        stream_message = getattr(response, "_telegram_message", None)
        is_draft = getattr(response, "_is_draft", False)
        chunks = self._split_text(response.content, 4096)

        # If draft mode: always send fresh (draft was cleared earlier)
        if is_draft:
            stream_message = None

        # Single chunk: edit stream message if possible
        if len(chunks) == 1 and len(chunks[0]) + len(footer) <= 4096:
            reply_text = chunks[0] + footer
            if stream_message:
                edited = False
                try:
                    await stream_message.edit_text(
                        text=reply_text,
                        parse_mode=ParseMode.MARKDOWN,
                        reply_markup=keyboard,
                    )
                    edited = True
                except TelegramBadRequest:
                    edited = await self._safe_edit_text(stream_message, reply_text, reply_markup=keyboard)
                if edited:
                    self._try_register_feedback_meta(stream_message, message, response, task_category, interaction_id)
                    return
                # Edit failed — delete stream msg and fall through to send fresh
                try:
                    await stream_message.delete()
                except Exception:
                    pass

            sent = await self._safe_answer(message, reply_text, parse_mode=ParseMode.MARKDOWN, reply_markup=keyboard)
            self._try_register_feedback_meta(sent, message, response, task_category, interaction_id)
            return

        # Multi-chunk: if we have a stream message, delete it and send all chunks fresh
        # to avoid showing the full streamed content then a chopped version
        if stream_message and len(chunks) > 1:
            try:
                await stream_message.delete()
            except Exception:
                pass
            stream_message = None

        for idx, chunk in enumerate(chunks):
            is_last = idx == len(chunks) - 1
            text = chunk + footer if is_last else chunk
            markup = keyboard if is_last else None
            sent = await self._safe_answer(
                message,
                text,
                parse_mode=ParseMode.MARKDOWN if not is_last else None,
                reply_markup=markup,
            )
            if is_last:
                self._try_register_feedback_meta(sent, message, response, task_category, interaction_id)

    async def _safe_send_chat_action(self, message: Message, action: ChatAction = ChatAction.TYPING):
        now = time.monotonic()
        action_key = (message.chat.id, str(action.value))
        last_sent_at = self._last_chat_action_at.get(action_key, 0.0)
        interval = getattr(self.config, "telegram_chat_action_interval", 4.0)
        if now - last_sent_at < interval:
            return

        try:
            await message.bot.send_chat_action(chat_id=message.chat.id, action=action)
            self._last_chat_action_at[action_key] = now
        except (TelegramNetworkError, TelegramRetryAfter) as e:
            logger.warning("Telegram typing action skipped: %s", e)

    async def _safe_send_prepared_file(self, message: Message, prepared_file: PreparedFile) -> bool:
        caption = prepared_file.caption or None
        try:
            action = ChatAction.UPLOAD_PHOTO if prepared_file.as_photo else ChatAction.UPLOAD_DOCUMENT
            await self._safe_send_chat_action(message, action=action)
            input_file = FSInputFile(prepared_file.path)
            if prepared_file.as_photo:
                try:
                    await message.answer_photo(photo=input_file, caption=caption)
                    return True
                except TelegramBadRequest as e:
                    logger.info("Telegram photo send failed, falling back to document: %s", e)
                    input_file = FSInputFile(prepared_file.path)
            await message.answer_document(document=input_file, caption=caption)
            return True
        except TelegramBadRequest as e:
            logger.warning("Telegram file send failed: %s", e)
            await self._safe_answer(
                message,
                f"ERROR\nGagal mengirim {prepared_file.display_path}: {str(e)[:220]}",
            )
            return False
        except (TelegramNetworkError, TelegramRetryAfter) as e:
            logger.warning("Telegram file send failed: %s", e)
            await self._safe_answer(message, f"ERROR\nGagal mengirim {prepared_file.display_path}: {e}")
            return False

    # ---- AI Autonomous Shell ----

    _SHELL_PATTERN = re.compile(r"\[\[shell:(.+?)\]\]", re.DOTALL)

    _AI_SHELL_BLOCKED = {
        "rm ", "rm -", "mv ", "chmod ", "chown ", "mkfs", "dd ",
        "git reset", "git checkout", "shutdown", "reboot", ":(){",
        "> /dev", "| sh", "| bash", "curl ", "wget ", "nc ", "ncat ",
        "python -c", "python3 -c", "perl -e", "ruby -e",
    }

    _AI_SHELL_ALLOWED_PREFIXES = (
        "ls", "cat", "pwd", "df", "du", "ps", "top", "free", "uptime",
        "whoami", "uname", "date", "echo", "head", "tail", "grep",
        "find", "wc", "sort", "uniq", "git status", "git log", "git diff",
        "git branch", "python -m py_compile", "pip list", "pip freeze",
        "nvidia-smi", "systemctl status", "journalctl", "docker ps",
        "docker images", "docker-compose ps", "netstat", "ss -tlnp",
    )

    def _is_ai_shell_safe(self, command: str) -> bool:
        lowered = command.lower().strip()
        if any(b in lowered for b in self._AI_SHELL_BLOCKED):
            return False
        if not any(lowered.startswith(p.lower()) for p in self._AI_SHELL_ALLOWED_PREFIXES):
            return False
        return True

    def _execute_ai_shell(self, command: str) -> tuple[bool, str]:
        try:
            completed = subprocess.run(
                command,
                cwd=Path(__file__).parent.parent,
                shell=True,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=self.config.ai_shell_timeout,
            )
            output = completed.stdout.strip()[:self.config.ai_shell_max_output]
            return completed.returncode == 0, output or "(no output)"
        except subprocess.TimeoutExpired:
            return False, f"Command timed out after {self.config.ai_shell_timeout}s"
        except OSError as e:
            return False, f"Failed to run command: {e}"

    async def _process_ai_shell_commands(
        self,
        response: BrainResponse,
        user_message: str,
        model: str,
        fallback_model: list[str],
        memory_context: str,
        conversation_history: list[dict],
        user_name: str | None,
        personality_context: str,
    ) -> BrainResponse:
        """Detect [[shell:cmd]] in AI response, execute, and feed result back."""
        if not self.config.enable_ai_shell:
            return response

        matches = self._SHELL_PATTERN.findall(response.content)
        if not matches:
            return response

        # Only allow 1 shell round per message to prevent loops
        command = matches[0].strip()
        if not self._is_ai_shell_safe(command):
            logger.warning("AI shell command blocked: %s", command[:100])
            cleaned = self._SHELL_PATTERN.sub(
                f"\n[Shell ditolak: command tidak aman]\n",
                response.content,
                count=1,
            )
            return BrainResponse(
                content=cleaned,
                model_used=response.model_used,
                response_time=response.response_time,
                tokens_used=response.tokens_used,
                cost_estimate=response.cost_estimate,
                success=response.success,
            )

        logger.info("🖥️ AI shell executing: %s", command[:120])
        ok, output = self._execute_ai_shell(command)
        status = "✅" if ok else "❌"
        result_block = f"\n\n[SHELL RESULT {status}]\n```\n{output}\n```\n"

        # Build follow-up prompt
        follow_up = (
            f"User: {user_message}\n"
            f"You ran: [[shell:{command}]]\n"
            f"Result:\n{output}\n\n"
            f"Now answer the user naturally based on this result."
        )

        follow_history = (conversation_history or []) + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": response.content + result_block},
            {"role": "user", "content": "Please answer based on the shell result above."},
        ]

        final = await self.brain.think(
            user_message=follow_up,
            model=model,
            fallback_model=fallback_model,
            memory_context=memory_context,
            conversation_history=follow_history[-10:],
            user_name=user_name,
            personality_context=personality_context,
        )

        # Preserve original model attribution but add shell indicator
        final_content = final.content
        if ok:
            final_content = f"{final_content}\n\n`🖥️ shell: {command[:40]}{'...' if len(command) > 40 else ''}`"
        return BrainResponse(
            content=final_content,
            model_used=final.model_used,
            response_time=round(response.response_time + final.response_time, 2),
            tokens_used=response.tokens_used + final.tokens_used,
            cost_estimate=round(response.cost_estimate + final.cost_estimate, 4),
            success=final.success,
        )

    async def _safe_answer(self, message: Message, text: str, **kwargs):
        try:
            return await message.answer(text, **kwargs)
        except TelegramBadRequest as e:
            # Fallback: if parse_mode or markdown failed, retry plain text
            if kwargs.get("parse_mode"):
                logger.warning("Telegram answer with parse_mode failed: %s", e)
                kwargs.pop("parse_mode", None)
                try:
                    return await message.answer(text, **kwargs)
                except (TelegramBadRequest, TelegramNetworkError, TelegramRetryAfter) as e2:
                    logger.warning("Telegram plain answer fallback failed: %s", e2)
                    return None
            logger.warning("Telegram answer failed: %s", e)
            return None
        except (TelegramNetworkError, TelegramRetryAfter) as e:
            logger.warning("Telegram answer failed: %s", e)
            return None

    async def _safe_edit_text(self, message: Message, text: str, **kwargs) -> bool:
        try:
            await message.edit_text(text=text, **kwargs)
            return True
        except (TelegramNetworkError, TelegramRetryAfter) as e:
            logger.warning("Telegram edit failed: %s", e)
            return False
        except TelegramBadRequest as e:
            logger.debug("Telegram edit skipped: %s", e)
            return False

    async def _safe_send_message_draft(self, chat_id: int, draft_id: int, text: str) -> bool | None:
        if not self.bot:
            return False
        if not text:
            return False

        now = time.monotonic()
        retry_until = self._draft_retry_until.get(chat_id, 0.0)
        if now < retry_until:
            return None

        try:
            result = await self.bot(
                SendMessageDraft(chat_id=chat_id, draft_id=draft_id, text=text)
            )
            self._draft_retry_until.pop(chat_id, None)
            return bool(result)
        except TelegramRetryAfter as e:
            retry_after = float(getattr(e, "retry_after", 1) or 1)
            self._draft_retry_until[chat_id] = time.monotonic() + retry_after
            logger.info("Telegram draft rate-limited; pausing draft updates for %.1fs", retry_after)
            return None
        except TelegramBadRequest as e:
            logger.warning("Telegram draft update failed: %s", e)
            return False
        except TelegramNetworkError as e:
            logger.warning("Telegram draft update failed: %s", e)
            return False

    @staticmethod
    def _split_text(text: str, max_len: int) -> list[str]:
        """Split text into chunks <= max_len, preferring paragraph/sentence boundaries."""
        if len(text) <= max_len:
            return [text]

        chunks: list[str] = []
        while text:
            if len(text) <= max_len:
                chunks.append(text)
                break

            chunk = text[:max_len]
            # Try to break at paragraph
            break_idx = chunk.rfind("\n\n")
            if break_idx > max_len // 4:
                chunks.append(text[:break_idx])
                text = text[break_idx:].lstrip("\n")
                continue

            # Try to break at newline
            break_idx = chunk.rfind("\n")
            if break_idx > max_len // 4:
                chunks.append(text[:break_idx])
                text = text[break_idx:].lstrip("\n")
                continue

            # Try to break at sentence end
            for sep in (". ", "! ", "? ", "; "):
                break_idx = chunk.rfind(sep)
                if break_idx > max_len // 4:
                    chunks.append(text[: break_idx + len(sep)])
                    text = text[break_idx + len(sep):]
                    break
            else:
                # Hard break at space
                break_idx = chunk.rfind(" ")
                if break_idx > max_len // 4:
                    chunks.append(text[:break_idx])
                    text = text[break_idx:].lstrip(" ")
                else:
                    # Absolute hard break
                    chunks.append(text[:max_len])
                    text = text[max_len:]
        return chunks

    @staticmethod
    def _stream_preview(content: str) -> str:
        if len(content) <= 3900:
            return content + " ▌"
        return content[-3900:] + " ▌"

    @staticmethod
    def _draft_preview(content: str) -> str:
        if len(content) <= 3900:
            return content
        return content[-3900:]

    @staticmethod
    def _format_relative_time(target: datetime) -> str:
        now = datetime.now(timezone.utc)
        diff = target - now
        if diff.total_seconds() <= 0:
            return "sekarang"
        minutes = int(diff.total_seconds() // 60)
        hours = int(diff.total_seconds() // 3600)
        days = int(diff.total_seconds() // 86400)
        if days > 0:
            return f"{days} hari lagi"
        if hours > 0:
            return f"{hours} jam lagi"
        return f"{minutes} menit lagi"

    async def _feedback_callback(self, callback: CallbackQuery):
        if not callback.message or not callback.from_user:
            await callback.answer("Pesan tidak ditemukan.", show_alert=True)
            return

        meta = self._feedback_registry.get(callback.message.message_id)
        if not meta:
            await callback.answer("Feedback window expired.", show_alert=True)
            return

        if callback.from_user.id != meta.user_id:
            await callback.answer("Tombol ini bukan untukmu.", show_alert=True)
            return

        if meta.responded:
            await callback.answer("Feedback sudah diterima. Terima kasih!", show_alert=True)
            return

        meta.responded = True
        is_positive = callback.data == "fb:like"

        try:
            await self.learner.process_feedback(
                user_id=meta.user_id,
                interaction_id=meta.interaction_id,
                feedback="positive" if is_positive else "negative",
                task_category=meta.task_category,
                model_used=meta.model_used,
            )

            if not is_positive and self.self_improver:
                self.self_improver.record_negative_feedback(
                    user_id=meta.user_id,
                    user_message=meta.user_message,
                    bot_response=meta.bot_response,
                    task_category=meta.task_category,
                    model_used=meta.model_used,
                )

            confirm_label = "✅ Terima kasih!" if is_positive else "📝 Catatan diterima."
            await callback.message.edit_reply_markup(
                reply_markup=InlineKeyboardMarkup(
                    inline_keyboard=[[InlineKeyboardButton(text=confirm_label, callback_data="fb:done")]]
                )
            )
            await callback.answer()
        except Exception as e:
            logger.error("Feedback handling failed: %s", e)
            await callback.answer("Gagal mencatat feedback. Coba lagi nanti.", show_alert=True)

    @staticmethod
    def _build_feedback_keyboard() -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(text="👍", callback_data="fb:like"),
                    InlineKeyboardButton(text="👎", callback_data="fb:dislike"),
                ]
            ]
        )

    def _try_register_feedback_meta(
        self,
        sent_message: Message | None,
        original_message: Message,
        response: BrainResponse,
        task_category: str,
        interaction_id: str | None,
    ):
        if not interaction_id or not sent_message:
            return
        self._register_feedback(
            sent_message.message_id,
            _FeedbackMeta(
                interaction_id=interaction_id,
                user_id=original_message.from_user.id if original_message.from_user else 0,
                user_message=original_message.text or "",
                bot_response=response.content,
                task_category=task_category,
                model_used=response.model_used,
            ),
        )

    def _register_feedback(self, message_id: int, meta: _FeedbackMeta):
        self._feedback_registry[message_id] = meta
        overflow = len(self._feedback_registry) - 200
        if overflow > 0:
            for key in list(self._feedback_registry.keys())[:overflow]:
                self._feedback_registry.pop(key, None)

    def _next_upload_path(self, message: Message, filename: str) -> Path:
        user_id = message.from_user.id if message.from_user else message.chat.id
        upload_dir = self.config.telegram_upload_dir / str(user_id)
        upload_dir.mkdir(parents=True, exist_ok=True)

        safe_name = self._safe_filename(filename)
        candidate = upload_dir / safe_name
        if not candidate.exists():
            return candidate

        stem = candidate.stem or "file"
        suffix = candidate.suffix
        for index in range(1, 1000):
            next_candidate = upload_dir / f"{stem}_{index}{suffix}"
            if not next_candidate.exists():
                return next_candidate
        return upload_dir / f"{stem}_{int(time.time())}{suffix}"

    async def _maybe_acknowledge_learned_profile(self, message: Message, user_message: str):
        """Send a subtle acknowledgment when profile info is detected in chat."""
        try:
            memories = self.learner._extract_direct_memories(user_message)
            if not memories:
                return
            # Only acknowledge identity/profile facts, not generic preferences
            profile_types = {"identity", "profile"}
            profile_memories = [m for m in memories if m.get("topic") in profile_types]
            if not profile_memories:
                return

            # Build a short acknowledgment
            labels = []
            for mem in profile_memories:
                fact_lower = mem.get("fact", "").lower()
                if "name is" in fact_lower:
                    labels.append("namamu")
                elif "called" in fact_lower or "panggil" in fact_lower:
                    labels.append("panggilanmu")
                elif "works as" in fact_lower or "is a" in fact_lower:
                    labels.append("pekerjaanmu")
                elif "interested in" in fact_lower:
                    labels.append("minatmu")
                else:
                    labels.append("itu")

            if labels:
                unique_labels = []
                seen = set()
                for label in labels:
                    if label not in seen:
                        unique_labels.append(label)
                        seen.add(label)
                ack = f"🧠 Oke, aku catat {', '.join(unique_labels)}!"
                await self._safe_answer(message, ack)
        except Exception:
            pass

    @staticmethod
    def _parse_intro_args(text: str) -> dict:
        """Parse /intro arguments into structured profile fields."""
        profile: dict[str, str] = {}
        text = text.strip()

        # Pattern 1: Pipe-separated key:value format
        # Nama: Budi | Job: Developer | Minat: AI, Coding | Preferensi: Python | Panggil: mas
        pipe_pattern = re.findall(
            r"(?i)(?:nama|panggil|job|pekerjaan|minat|interests?|preferensi|preferences?)\s*[:=]\s*([^|]+)",
            text,
        )
        if pipe_pattern:
            # Try to match each field individually
            for key, val in re.findall(
                r"(?i)(nama|panggil|job|pekerjaan|minat|interests?|preferensi|preferences?)\s*[:=]\s*([^|]+)",
                text,
            ):
                key_lower = key.lower().strip()
                val_clean = val.strip(" .,;!\n")
                if key_lower in {"nama", "name"}:
                    profile["nama"] = val_clean
                elif key_lower in {"panggil", "call", "nickname"}:
                    profile["panggil"] = val_clean
                elif key_lower in {"job", "pekerjaan", "work", "profesi"}:
                    profile["job"] = val_clean
                elif key_lower in {"minat", "interests", "hobi", "hobby"}:
                    profile["minat"] = val_clean
                elif key_lower in {"preferensi", "preferences", "suka", "like"}:
                    profile["preferensi"] = val_clean
            return profile

        # Pattern 2: Natural language extraction
        text_lower = text.lower()

        # Name: "nama saya/gue/aku ..." or "name is ..."
        m = re.search(r"(?:nama\s+(?:saya|gue|aku)|my name is|nama:)\s+([^,.\n!?]+)", text, flags=re.IGNORECASE)
        if m:
            profile["nama"] = m.group(1).strip()

        # Nickname: "panggil saya/gue/aku ..." or "call me ..."
        m = re.search(r"(?:panggil\s+(?:saya|gue|aku)|call me|panggil:)\s+([^,.\n!?]+)", text, flags=re.IGNORECASE)
        if m:
            profile["panggil"] = m.group(1).strip()

        # Job: "developer", "saya seorang ...", "kerja sebagai ...", "job: ..."
        m = re.search(r"(?:saya seorang|aku seorang|i am a[n]?|i work as|pekerjaan\s*[:=]|job\s*[:=]|kerja\s+sebagai|profesi\s*[:=])\s+([^,.\n!?]+)", text, flags=re.IGNORECASE)
        if m:
            profile["job"] = m.group(1).strip()
        else:
            # Fallback: common job titles
            job_keywords = r"\b(developer|engineer|programmer|designer|student|mahasiswa|pelajar|data scientist|manager|freelancer|fullstack|backend|frontend|devops)\b"
            m = re.search(job_keywords, text, flags=re.IGNORECASE)
            if m:
                profile["job"] = m.group(1).strip()

        # Interests: "suka ...", "minat ...", "interested in ..."
        m = re.search(r"(?:suka|minat|interested in|hobi|hobi\s*[:=]|minat\s*[:=])\s+([^,.\n!?]+)", text, flags=re.IGNORECASE)
        if m:
            profile["minat"] = m.group(1).strip()

        # Preferences: "preferensi ...", "prefer ...", "suka pakai ..."
        m = re.search(r"(?:preferensi|prefer|suka\s+pakai|preferensi\s*[:=])\s+([^,.\n!?]+)", text, flags=re.IGNORECASE)
        if m:
            profile["preferensi"] = m.group(1).strip()

        return profile

    @staticmethod
    def _safe_filename(filename: str) -> str:
        cleaned = Path(filename).name.strip()
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", cleaned)
        cleaned = cleaned.strip("._")
        return cleaned[:120] or "upload.bin"

    @staticmethod
    def _telegram_text_limit(text: str, limit: int = 3900) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 24] + "\n...[dipotong]"

    @staticmethod
    def _command_args(message: Message) -> str:
        text = message.text or ""
        parts = text.split(maxsplit=1)
        return parts[1].strip() if len(parts) > 1 else ""

    def _authorized_message(self, message: Message) -> bool:
        return self._is_message_authorized(message)

    def _target_group_id(self, message: Message) -> int | None:
        args = self._command_args(message)
        match = re.search(r"-?\d{5,20}", args)
        if match:
            return int(match.group(0))
        if message.chat.type in {ChatType.GROUP, ChatType.SUPERGROUP}:
            return message.chat.id
        return None
