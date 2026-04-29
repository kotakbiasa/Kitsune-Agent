"""
Kitsune local tool registry.

Safe local tools for explicit Telegram /tool calls.
"""

from __future__ import annotations

import fnmatch
import json
import mimetypes
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_MAX_SEND_FILE_BYTES = 45 * 1024 * 1024
TEXT_PREVIEW_BYTES = 64 * 1024
TEXT_PREVIEW_CHARS = 3500
TEXT_EXTENSIONS = {
    ".bat",
    ".cfg",
    ".conf",
    ".csv",
    ".css",
    ".env",
    ".gitignore",
    ".go",
    ".html",
    ".ini",
    ".java",
    ".js",
    ".json",
    ".kt",
    ".log",
    ".md",
    ".php",
    ".properties",
    ".py",
    ".rb",
    ".rs",
    ".sh",
    ".sql",
    ".swift",
    ".toml",
    ".ts",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}


@dataclass(frozen=True)
class PreparedFile:
    path: Path
    display_path: str
    caption: str = ""
    as_photo: bool = False
    mime_type: str = ""
    size_bytes: int = 0


@dataclass(frozen=True)
class FileInspection:
    path: Path
    display_path: str
    size_bytes: int
    mime_type: str
    kind: str
    is_text: bool
    preview: str = ""
    truncated: bool = False

    def render(self) -> str:
        lines = [
            f"File: {self.display_path}",
            f"Jenis: {self.kind}",
            f"MIME: {self.mime_type}",
            f"Ukuran: {format_size(self.size_bytes)}",
        ]
        if self.preview:
            suffix = "\n...[preview dipotong]" if self.truncated else ""
            lines.extend(["", "Preview:", f"```text\n{self.preview}{suffix}\n```"])
        elif not self.is_text:
            lines.append("")
            lines.append("Preview teks tidak tersedia untuk file binary/dokumen ini.")
        return "\n".join(lines)


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    output: str
    files: tuple[PreparedFile, ...] = ()


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    purpose: str
    usage: str


class LocalToolRegistry:
    """Safe local tools for explicit Telegram /tool calls."""

    def __init__(
        self,
        workspace_root: str | Path = PROJECT_ROOT,
        enable_shell: bool = False,
        max_send_file_bytes: int | None = None,
    ):
        self.workspace_root = Path(workspace_root).resolve()
        self.enable_shell = enable_shell
        self.max_send_file_bytes = max_send_file_bytes or max_send_file_bytes_from_env()
        self._definitions = {
            "list_files": ToolDefinition(
                name="list_files",
                purpose="List files under the Kitsune workspace",
                usage="/tool list_files [path] [glob]",
            ),
            "read_file": ToolDefinition(
                name="read_file",
                purpose="Read a text file from the Kitsune workspace",
                usage="/tool read_file <path> [offset] [limit]",
            ),
            "file_info": ToolDefinition(
                name="file_info",
                purpose="Detect file type, MIME, size, and text preview when possible",
                usage="/tool file_info <path>",
            ),
            "grep": ToolDefinition(
                name="grep",
                purpose="Search text files with a regex pattern",
                usage="/tool grep <pattern> [path] [glob]",
            ),
            "send_file": ToolDefinition(
                name="send_file",
                purpose="Send a workspace file back to this Telegram chat",
                usage="/tool send_file <path> [caption]",
            ),
            "send_photo": ToolDefinition(
                name="send_photo",
                purpose="Send an image file as a native Telegram photo",
                usage="/tool send_photo <path> [caption]",
            ),
            "shell": ToolDefinition(
                name="shell",
                purpose="Run a shell command in the Kitsune workspace when explicitly enabled",
                usage="/tool shell <command>",
            ),
            "claude": ToolDefinition(
                name="claude",
                purpose="Run safe local equivalents for selected claude-code tool names",
                usage="/tool claude <FileReadTool|GlobTool|GrepTool|BashTool> <args>",
            ),
        }

    def describe(self) -> str:
        """Return a user-readable tool list."""
        lines = ["Tools aktif:"]
        for tool in self._definitions.values():
            status = "disabled" if tool.name == "shell" and not self.enable_shell else "enabled"
            lines.append(f"- {tool.name} ({status}): {tool.purpose}")
            lines.append(f"  {tool.usage}")
        lines.append(f"Workspace root: {self.workspace_root}")
        lines.append(f"Max send_file size: {self.max_send_file_bytes // (1024 * 1024)} MB")
        return "\n".join(lines)

    def run(self, raw_args: str) -> ToolResult:
        """Dispatch a /tool command string."""
        try:
            args = shlex.split(raw_args)
        except ValueError as e:
            return ToolResult(False, f"Argumen tidak valid: {e}")

        if not args:
            return ToolResult(False, self.describe())

        name, rest = args[0], args[1:]
        if name == "list_files":
            return self._list_files(rest)
        if name == "read_file":
            return self._read_file(rest)
        if name in {"file_info", "file_type", "inspect_file"}:
            return self._file_info(rest)
        if name == "grep":
            return self._grep(rest)
        if name == "send_file":
            return self._prepare_send_file(rest, as_photo=False)
        if name in {"send_photo", "send_image"}:
            return self._prepare_send_file(rest, as_photo=True)
        if name == "shell":
            return self._shell(rest)
        if name == "claude":
            return self._run_claude_alias(rest)

        return ToolResult(False, f"Tool tidak dikenal: {name}\n\n{self.describe()}")

    def _run_claude_alias(self, args: list[str]) -> ToolResult:
        if not args:
            return ToolResult(
                False,
                "Usage: /tool claude <FileReadTool|GlobTool|GrepTool|BashTool> <args>",
            )

        source_tool, rest = args[0], args[1:]
        aliases = {
            "filereadtool": self._read_file,
            "globtool": self._list_files,
            "greptool": self._grep,
            "bashtool": self._shell,
            "sendmessagetool": self._prepare_send_message_alias,
        }
        runner = aliases.get(source_tool.lower())
        if runner is None:
            return ToolResult(False, f"Tool tidak dikenal: {source_tool}")

        result = runner(rest)
        return ToolResult(result.ok, result.output, result.files)

    def _prepare_send_message_alias(self, args: list[str]) -> ToolResult:
        if args and args[0] in {"file", "document", "photo", "image"}:
            as_photo = args[0] in {"photo", "image"}
            return self._prepare_send_file(args[1:], as_photo=as_photo)
        return ToolResult(
            False,
            "Usage: /tool claude SendMessageTool <file|document|photo|image> <path> [caption]",
        )

    def _list_files(self, args: list[str]) -> ToolResult:
        raw_path = args[0] if args else "."
        glob_pattern = args[1] if len(args) > 1 else "*"

        try:
            base = self._resolve_workspace_path(raw_path)
        except ValueError as e:
            return ToolResult(False, str(e))

        if not base.exists():
            return ToolResult(False, f"Path tidak ditemukan: {raw_path}")

        files = []
        if base.is_file():
            files = [base]
        else:
            for path in base.rglob("*"):
                if path.is_file() and fnmatch.fnmatch(path.name, glob_pattern):
                    files.append(path)
                if len(files) >= 80:
                    break

        if not files:
            return ToolResult(True, "Tidak ada file yang cocok.")

        lines = [str(path.relative_to(self.workspace_root)) for path in sorted(files)]
        return ToolResult(True, "\n".join(lines))

    def _read_file(self, args: list[str]) -> ToolResult:
        if not args:
            return ToolResult(False, "Usage: /tool read_file <path> [offset] [limit]")

        try:
            path = self._resolve_workspace_path(args[0])
        except ValueError as e:
            return ToolResult(False, str(e))

        offset = self._parse_int(args[1], 0) if len(args) > 1 else 0
        limit = self._parse_int(args[2], 120) if len(args) > 2 else 120
        limit = max(1, min(limit, 300))

        if not path.is_file():
            return ToolResult(False, f"Bukan file: {args[0]}")

        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as e:
            return ToolResult(False, f"Gagal membaca file: {e}")

        selected = lines[offset : offset + limit]
        numbered = [f"{offset + idx + 1}: {line}" for idx, line in enumerate(selected)]
        return ToolResult(True, "\n".join(numbered) or "(file kosong)")

    def _file_info(self, args: list[str]) -> ToolResult:
        if not args:
            return ToolResult(False, "Usage: /tool file_info <path>")

        try:
            path = self._resolve_workspace_path(args[0])
        except ValueError as e:
            return ToolResult(False, str(e))

        if not path.is_file():
            return ToolResult(False, f"Bukan file: {args[0]}")
        if self._is_sensitive_path(path):
            return ToolResult(False, "File ditolak karena terlihat berisi secret atau data runtime sensitif.")

        try:
            inspection = inspect_file(path, str(path.relative_to(self.workspace_root)))
        except OSError as e:
            return ToolResult(False, f"Gagal membaca file: {e}")
        return ToolResult(True, inspection.render())

    def _grep(self, args: list[str]) -> ToolResult:
        if not args:
            return ToolResult(False, "Usage: /tool grep <pattern> [path] [glob]")

        pattern = args[0]
        raw_path = args[1] if len(args) > 1 else "."
        glob_pattern = args[2] if len(args) > 2 else "*.py"

        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return ToolResult(False, f"Regex tidak valid: {e}")

        try:
            base = self._resolve_workspace_path(raw_path)
        except ValueError as e:
            return ToolResult(False, str(e))

        candidates = [base] if base.is_file() else base.rglob("*")
        matches = []
        for path in candidates:
            if not path.is_file() or not fnmatch.fnmatch(path.name, glob_pattern):
                continue
            try:
                for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if regex.search(line):
                        rel = path.relative_to(self.workspace_root)
                        matches.append(f"{rel}:{line_no}: {line[:220]}")
                        if len(matches) >= 80:
                            return ToolResult(True, "\n".join(matches))
            except OSError:
                continue

        return ToolResult(True, "\n".join(matches) if matches else "Tidak ada match.")

    def _prepare_send_file(self, args: list[str], as_photo: bool = False) -> ToolResult:
        if not args:
            command = "send_photo" if as_photo else "send_file"
            return ToolResult(False, f"Usage: /tool {command} <path> [caption]")

        raw_path = args[0]
        try:
            path = self._resolve_workspace_path(raw_path)
        except ValueError as e:
            return ToolResult(False, str(e))

        if not path.is_file():
            return ToolResult(False, f"Bukan file: {raw_path}")
        if self._is_sensitive_path(path):
            return ToolResult(False, "File ditolak karena terlihat berisi secret atau data runtime sensitif.")

        try:
            size_bytes = path.stat().st_size
        except OSError as e:
            return ToolResult(False, f"Gagal membaca metadata file: {e}")

        if size_bytes <= 0:
            return ToolResult(False, "File kosong, tidak dikirim.")
        if size_bytes > self.max_send_file_bytes:
            limit_mb = self.max_send_file_bytes // (1024 * 1024)
            size_mb = size_bytes / (1024 * 1024)
            return ToolResult(False, f"File terlalu besar: {size_mb:.1f} MB. Limit saat ini {limit_mb} MB.")

        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if as_photo and not mime_type.startswith("image/"):
            return ToolResult(False, f"send_photo hanya menerima image/*, terdeteksi: {mime_type}")

        display_path = str(path.relative_to(self.workspace_root))
        caption = " ".join(args[1:]).strip()
        prepared = PreparedFile(
            path=path,
            display_path=display_path,
            caption=caption[:1024],
            as_photo=as_photo,
            mime_type=mime_type,
            size_bytes=size_bytes,
        )
        mode = "photo" if as_photo else "document"
        return ToolResult(
            True,
            f"Siap mengirim {display_path} sebagai {mode} ({format_size(size_bytes)}).",
            (prepared,),
        )

    def _shell(self, args: list[str]) -> ToolResult:
        if not self.enable_shell:
            return ToolResult(
                False,
                "Tool shell belum aktif. Set ENABLE_SHELL_TOOL=true di .env kalau benar-benar diperlukan.",
            )
        if not args:
            return ToolResult(False, "Usage: /tool shell <command>")

        command = " ".join(args)
        if self._looks_destructive(command):
            return ToolResult(False, "Command ditolak karena terlihat destruktif.")

        try:
            completed = subprocess.run(
                command,
                cwd=self.workspace_root,
                shell=True,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=20,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(False, "Command timeout setelah 20 detik.")
        except OSError as e:
            return ToolResult(False, f"Gagal menjalankan command: {e}")

        output = completed.stdout.strip()[:6000]
        return ToolResult(completed.returncode == 0, output or f"Exit code: {completed.returncode}")

    def _resolve_workspace_path(self, raw_path: str) -> Path:
        path = (self.workspace_root / raw_path).resolve()
        if path != self.workspace_root and self.workspace_root not in path.parents:
            raise ValueError("Path ditolak karena keluar dari workspace Kitsune-Agent.")
        return path

    def _is_sensitive_path(self, path: Path) -> bool:
        rel_parts = path.relative_to(self.workspace_root).parts
        blocked_dirs = {".git", ".venv", "__pycache__", "memory_db"}
        if any(part in blocked_dirs for part in rel_parts):
            return True

        name = path.name.lower()
        blocked_names = {".env", ".env.local", ".env.production", "id_rsa", "id_ed25519"}
        blocked_suffixes = {".pem", ".key", ".p12", ".pfx"}
        return name in blocked_names or any(name.endswith(suffix) for suffix in blocked_suffixes)

    @staticmethod
    def _parse_int(raw_value: str, default: int) -> int:
        try:
            return int(raw_value)
        except ValueError:
            return default

    @staticmethod
    def _looks_destructive(command: str) -> bool:
        blocked = [
            "rm ",
            "rm -",
            "mv ",
            "chmod ",
            "chown ",
            "mkfs",
            "dd ",
            "git reset",
            "git checkout",
            "shutdown",
            "reboot",
            ":(){",
        ]
        lowered = command.lower()
        return any(token in lowered for token in blocked)


def tools_enabled_from_env() -> bool:
    return os.getenv("ENABLE_LOCAL_TOOLS", "").strip().lower() in {"1", "true", "yes", "on"}


def shell_enabled_from_env() -> bool:
    return os.getenv("ENABLE_SHELL_TOOL", "").strip().lower() in {"1", "true", "yes", "on"}


def max_send_file_bytes_from_env() -> int:
    raw_value = os.getenv("MAX_SEND_FILE_MB", "").strip()
    if not raw_value:
        return DEFAULT_MAX_SEND_FILE_BYTES
    try:
        mb = int(raw_value)
    except ValueError:
        return DEFAULT_MAX_SEND_FILE_BYTES
    return max(1, min(mb, 45)) * 1024 * 1024


def inspect_file(path: Path, display_path: str | None = None) -> FileInspection:
    size_bytes = path.stat().st_size
    mime_type = mimetypes.guess_type(path.name)[0] or _sniff_mime(path) or "application/octet-stream"
    kind = _kind_from_mime_or_suffix(mime_type, path.suffix.lower())
    is_text = _looks_like_text(path, mime_type)
    preview = ""
    truncated = False

    if is_text and size_bytes > 0:
        raw = path.read_bytes()[:TEXT_PREVIEW_BYTES]
        text = raw.decode("utf-8", errors="replace")
        preview = text[:TEXT_PREVIEW_CHARS]
        truncated = len(text) > TEXT_PREVIEW_CHARS or size_bytes > TEXT_PREVIEW_BYTES

    return FileInspection(
        path=path,
        display_path=display_path or path.name,
        size_bytes=size_bytes,
        mime_type=mime_type,
        kind=kind,
        is_text=is_text,
        preview=preview,
        truncated=truncated,
    )


def format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def _looks_like_text(path: Path, mime_type: str) -> bool:
    if mime_type.startswith("text/") or path.suffix.lower() in TEXT_EXTENSIONS:
        return True
    try:
        sample = path.read_bytes()[:4096]
    except OSError:
        return False
    if b"\x00" in sample:
        return False
    if not sample:
        return True
    decoded = sample.decode("utf-8", errors="replace")
    replacement_ratio = decoded.count("\ufffd") / max(len(decoded), 1)
    return replacement_ratio < 0.05


def _sniff_mime(path: Path) -> str | None:
    try:
        header = path.read_bytes()[:16]
    except OSError:
        return None
    if header.startswith(b"%PDF"):
        return "application/pdf"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header.startswith(b"GIF87a") or header.startswith(b"GIF89a"):
        return "image/gif"
    if header.startswith(b"PK\x03\x04"):
        return "application/zip"
    return None


def _kind_from_mime_or_suffix(mime_type: str, suffix: str) -> str:
    suffix_labels = {
        ".docx": "Word document",
        ".xlsx": "Excel spreadsheet",
        ".pptx": "PowerPoint presentation",
        ".pdf": "PDF document",
        ".zip": "ZIP archive",
    }
    if suffix in suffix_labels:
        return suffix_labels[suffix]
    if suffix in TEXT_EXTENSIONS:
        return "Text file"
    if mime_type.startswith("text/"):
        return "Text file"
    if mime_type.startswith("image/"):
        return "Image"
    if mime_type.startswith("audio/"):
        return "Audio"
    if mime_type.startswith("video/"):
        return "Video"
    if mime_type == "application/json":
        return "JSON document"
    if mime_type == "application/pdf":
        return "PDF document"
    if mime_type in {"application/zip", "application/x-zip-compressed"}:
        return "ZIP archive"
    return "Binary/document file"
