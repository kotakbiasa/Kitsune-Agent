"""
Kitsune Telegram Bot using aiogram.
Connects all components: Router, Brain, Memory, Learner, and local tools.
"""

from __future__ import annotations

import asyncio
import html
import logging
import re
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

from kitsune.brain import Brain, BrainResponse
from kitsune.config import Config
from kitsune.learner import Learner
from kitsune.memory import MemorySystem
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

        self._chat_history: dict[tuple[int, int], list[dict]] = {}
        self._draft_retry_until: dict[int, float] = {}
        self._last_chat_action_at: dict[tuple[int, str], float] = {}
        self._feedback_registry: dict[int, _FeedbackMeta] = {}

        self.bot: Bot | None = None
        self.dp: Dispatcher | None = None

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

        router = AiogramRouter()
        self._register_handlers(router)
        self.dp.include_router(router)

        self.reminders.start()
        logger.info("⏰ Reminder scheduler started.")
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
        router.message.register(
            self._handle_message,
            F.text,
            lambda message: bool(message.text and not message.text.startswith("/")),
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
            "• Kirim document/photo untuk cek jenis file dan membaca preview teks"
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
            history = self._chat_history.get(hist_key, [])
            response = await self.brain.think(
                user_message=file_prompt,
                model=primary_model,
                fallback_model=fallback_model,
                memory_context=memory_context,
                conversation_history=history,
                user_name=user_name,
            )

            self._chat_history.setdefault(hist_key, [])
            self._chat_history[hist_key].append({"role": "user", "content": f"[file] {inspection.display_path}\n{request}"})
            self._chat_history[hist_key].append({"role": "assistant", "content": response.content})
            if len(self._chat_history[hist_key]) > 20:
                self._chat_history[hist_key] = self._chat_history[hist_key][-20:]

            await self._send_or_finalize_response(message, response, task_category, interaction_id)

            learning_args = dict(
                user_id=user.id,
                user_message=f"[uploaded_file] {inspection.display_path}\n{request}\n{file_context[:1200]}",
                bot_response=response.content,
                task_category=task_category,
                model_used=response.model_used,
                response_time=response.response_time,
                response_success=self.learner.assess_response_success(response.content, response.success),
                interaction_id=interaction_id,
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

        interaction_id = f"int_{uuid.uuid4().hex[:12]}"
        hist_key = self._history_key(message)
        user_name = self._get_user_name(user.id)
        logger.info("📩 Message from %s (%d): %s", user.first_name, user.id, text[:100])

        try:
            await self._safe_send_chat_action(message)

            task_category, _ = await self.router.classify_task(text)
            primary_model, fallback_model = self.router.get_model_for_task(task_category)

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
                )
            else:
                response = await self.brain.think(
                    user_message=text,
                    model=primary_model,
                    fallback_model=fallback_model,
                    memory_context=memory_context,
                    conversation_history=history,
                    user_name=user_name,
                )

            self._chat_history.setdefault(hist_key, [])
            self._chat_history[hist_key].append({"role": "user", "content": text})
            self._chat_history[hist_key].append({"role": "assistant", "content": response.content})
            if len(self._chat_history[hist_key]) > 20:
                self._chat_history[hist_key] = self._chat_history[hist_key][-20:]

            await self._send_or_finalize_response(message, response, task_category, interaction_id)

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
            )
            if self.config.background_learning:
                asyncio.create_task(self._learn_after_response(**learning_args))
            else:
                await self._learn_after_response(**learning_args)

        except Exception as e:
            logger.error("Error handling message: %s\n%s", e, traceback.format_exc())
            await self._safe_answer(
                message,
                "😅 Maaf, terjadi error. Coba lagi ya!\n"
                f"Error: `{html.escape(str(e)[:200])}`",
                parse_mode=ParseMode.MARKDOWN,
            )

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
        logger.error("Unhandled aiogram exception: %s", event.exception, exc_info=event.exception)

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
            )
            if response:
                return response

        stream_message = await self._safe_answer(message, "...")
        if not stream_message:
            return await self.brain.think(user_message, model, fallback_model, memory_context, conversation_history, user_name)

        final_response = None
        content = ""
        last_edit_at = 0.0
        last_edit_len = 0

        async for event in self.brain.stream_think(user_message, model, fallback_model, memory_context, conversation_history, user_name):
            if event.get("type") == "delta":
                content += event.get("content", "")
                now = time.monotonic()
                if (
                    now - last_edit_at >= self.config.stream_edit_interval
                    and len(content) - last_edit_len >= self.config.stream_min_chars
                ):
                    if await self._safe_edit_text(stream_message, self._stream_preview(content)):
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
    ) -> BrainResponse | None:
        chat_id = message.chat.id
        draft_id = int(time.time() * 1000)
        final_response = None
        content = ""
        last_draft_at = 0.0
        last_draft_len = 0

        await self._safe_send_chat_action(message)

        async for event in self.brain.stream_think(user_message, model, fallback_model, memory_context, conversation_history, user_name):
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
        reply_text = response.content + footer
        keyboard = self._build_feedback_keyboard()

        stream_message = getattr(response, "_telegram_message", None)
        if stream_message:
            try:
                await stream_message.edit_text(
                    text=reply_text,
                    parse_mode=ParseMode.MARKDOWN,
                    reply_markup=keyboard,
                )
                self._try_register_feedback_meta(stream_message, message, response, task_category, interaction_id)
                return
            except TelegramBadRequest:
                await self._safe_edit_text(stream_message, reply_text, reply_markup=keyboard)
                self._try_register_feedback_meta(stream_message, message, response, task_category, interaction_id)
                return

        try:
            sent = await self._safe_answer(message, reply_text, parse_mode=ParseMode.MARKDOWN, reply_markup=keyboard)
            self._try_register_feedback_meta(sent, message, response, task_category, interaction_id)
        except TelegramBadRequest:
            sent = await self._safe_answer(message, reply_text, reply_markup=keyboard)
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

    async def _safe_answer(self, message: Message, text: str, **kwargs):
        try:
            return await message.answer(text, **kwargs)
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
