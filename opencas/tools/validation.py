"""Tool validation pipeline for OpenCAS.

Composable validators run before tool execution to enforce safety policy.
"""

from __future__ import annotations

import dataclasses
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclasses.dataclass
class ToolValidationResult:
    """Result of a tool validation check."""

    allowed: bool
    reason: Optional[str] = None
    warnings: List[str] = dataclasses.field(default_factory=list)
    resolved_path: Optional[str] = None
    command_permission_class: Optional[str] = None
    command_family: Optional[str] = None


@dataclasses.dataclass
class ToolValidationContext:
    """Context passed to validators."""

    roots: List[str] = dataclasses.field(default_factory=list)
    allow_any_read: bool = False
    max_read_bytes: Optional[int] = None
    max_write_bytes: Optional[int] = None


@dataclasses.dataclass(frozen=True)
class CommandAssessment:
    """Structured view of a shell command for policy and approval decisions."""

    family: str
    permission_class: str
    executable: Optional[str] = None
    subcommand: Optional[str] = None
    warnings: List[str] = dataclasses.field(default_factory=list)


class ToolValidator(ABC):
    """Base class for a single validation stage."""

    @abstractmethod
    def validate(
        self,
        tool_name: str,
        args: Dict[str, Any],
        context: ToolValidationContext,
    ) -> Optional[ToolValidationResult]:
        """Return a validation result, or None to abstain."""
        ...


class ToolValidationPipeline:
    """Runs a chain of validators before tool execution."""

    def __init__(
        self,
        validators: List[ToolValidator],
        default_context: Optional[ToolValidationContext] = None,
    ) -> None:
        self.validators = validators
        self.default_context = default_context or ToolValidationContext()

    def validate(
        self,
        tool_name: str,
        args: Dict[str, Any],
        context: Optional[ToolValidationContext] = None,
    ) -> ToolValidationResult:
        """Evaluate all validators and return the combined result."""
        ctx = self._merge_context(context)
        warnings: List[str] = []
        command_permission_class: Optional[str] = None
        command_family: Optional[str] = None

        for validator in self.validators:
            result = validator.validate(tool_name, args, ctx)
            if result is None:
                continue
            warnings.extend(result.warnings)
            if result.command_permission_class:
                command_permission_class = result.command_permission_class
            if result.command_family:
                command_family = result.command_family
            if not result.allowed:
                return ToolValidationResult(
                    allowed=False,
                    reason=result.reason,
                    warnings=warnings,
                    command_permission_class=command_permission_class,
                    command_family=command_family,
                )

        return ToolValidationResult(
            allowed=True,
            warnings=warnings,
            resolved_path=None,
            command_permission_class=command_permission_class,
            command_family=command_family,
        )

    def _merge_context(self, context: Optional[ToolValidationContext]) -> ToolValidationContext:
        if context is None:
            return self.default_context
        return ToolValidationContext(
            roots=context.roots if context.roots else self.default_context.roots,
            allow_any_read=context.allow_any_read or self.default_context.allow_any_read,
            max_read_bytes=context.max_read_bytes or self.default_context.max_read_bytes,
            max_write_bytes=context.max_write_bytes or self.default_context.max_write_bytes,
        )


def create_default_tool_validation_pipeline(
    roots: Optional[List[str]] = None,
    max_write_bytes: Optional[int] = None,
) -> ToolValidationPipeline:
    """Factory for the standard OpenCAS validation pipeline."""
    default_context = ToolValidationContext(
        roots=roots or [],
        max_write_bytes=max_write_bytes,
    )
    return ToolValidationPipeline(
        [
            CommandSafetyValidator(),
            FilesystemPathValidator(),
            FilesystemWatchlistValidator(),
            ContentSizeValidator(max_write_bytes=max_write_bytes),
        ],
        default_context=default_context,
    )


class SmartCommandValidator(ToolValidator):
    """
    Intelligent command safety validator utilizing a 'grey list'.
    Analyzes commands, breaks them down into sub-components, and checks safety
    ratings (general use vs task-specific use). Unrecognized or low-confidence
    commands default to 'caution' (blocking or warning) to learn over time.
    """
    
    def __init__(self, grey_list_path: str = ".opencas_command_greylist.json") -> None:
        self.grey_list_path = grey_list_path
        self.greylist = self._load_greylist()

    def _load_greylist(self) -> dict:
        import json, os
        if os.path.exists(self.grey_list_path):
            try:
                with open(self.grey_list_path, 'r') as f:
                    return json.load(f)
            except Exception:
                return self._default_greylist()
        return self._default_greylist()
        
    def _default_greylist(self) -> dict:
        return {
            "echo": {"general_safety": 0.0, "confidence": 1.0, "task_specific": {}},
            "ls": {"general_safety": 0.0, "confidence": 1.0, "task_specific": {}},
            "pwd": {"general_safety": 0.0, "confidence": 1.0, "task_specific": {}},
            "cat": {"general_safety": 0.1, "confidence": 1.0, "task_specific": {}},
            "pytest": {"general_safety": 0.1, "confidence": 0.9, "task_specific": {}},
            "git": {"general_safety": 0.2, "confidence": 0.9, "task_specific": {}},
            "python": {"general_safety": 0.2, "confidence": 0.9, "task_specific": {}},
            "python3": {"general_safety": 0.2, "confidence": 0.9, "task_specific": {}},
            "pip": {"general_safety": 0.3, "confidence": 0.9, "task_specific": {}},
            "rm": {"general_safety": 0.8, "confidence": 0.9, "task_specific": {}},
            "sudo": {"general_safety": 0.9, "confidence": 0.9, "task_specific": {}},
        }

    def _save_greylist(self) -> None:
        import json
        try:
            with open(self.grey_list_path, 'w') as f:
                json.dump(self.greylist, f, indent=2)
        except Exception:
            pass

    def _break_down_command(self, command: str) -> List[str]:
        """Breaks down a command into base executables without executing them."""
        import re, shlex
        # Split by logical operators, pipes, and command substitution sequences
        parts = re.split(r'\||&&|\|\||;|\$\(|\`', command)
        sub_commands = []
        for part in parts:
            part = part.replace(')', '').replace('`', '').strip()
            if not part: continue
            try:
                tokens = shlex.split(part)
            except ValueError:
                tokens = part.split()
            if not tokens: continue
            
            first = tokens[0].lower()
            if first in ("sh", "bash", "zsh", "eval", "sudo") and len(tokens) > 1:
                if "-c" in tokens:
                    idx = tokens.index("-c")
                    if idx + 1 < len(tokens):
                        sub_commands.extend(self._break_down_command(tokens[idx + 1]))
                else:
                    sub_commands.append(" ".join(tokens[1:]))
            else:
                sub_commands.append(first)
        return sub_commands

    def validate(
        self,
        tool_name: str,
        args: Dict[str, Any],
        context: ToolValidationContext,
    ) -> Optional[ToolValidationResult]:
        if tool_name != "bash_run_command":
            return None
        
        command = str(args.get("command", "")).strip()
        if not command:
            return ToolValidationResult(allowed=False, reason="missing command")

        sub_commands = self._break_down_command(command)
        
        warnings = []
        overall_allowed = True
        command_family = "safe"

        for base_cmd in sub_commands:
            base_cmd = base_cmd.split()[0].lower() if " " in base_cmd else base_cmd.lower()
            
            if base_cmd not in self.greylist:
                # Add to greylist with caution state to learn over time
                self.greylist[base_cmd] = {
                    "general_safety": 0.6, # Default to slightly unsafe/unknown
                    "confidence": 0.1,     # Low confidence, requires research
                    "task_specific": {}
                }
                self._save_greylist()
                warnings.append(f"Unrecognized command '{base_cmd}'. Defaulting to caution. Please assess safety and update greylist.")
                overall_allowed = False
                command_family = "unknown"
                break
            
            entry = self.greylist.get(base_cmd, {})
            safety = entry.get("general_safety", 0.5)
            conf = entry.get("confidence", 0.1)

            if safety > 0.7:
                overall_allowed = False
                command_family = "filesystem_destructive" if "rm" in base_cmd else "high_risk"
                warnings.append(f"Command '{base_cmd}' exceeds safety threshold ({safety}).")
                break
            elif safety > 0.4 and conf < 0.5:
                warnings.append(f"Low confidence on safety of '{base_cmd}'. Assessed safety: {safety}")
                overall_allowed = False
                command_family = "caution_required"
                break

        if not overall_allowed:
            return ToolValidationResult(
                allowed=False,
                reason=f"Command failed smart validation: " + " ".join(warnings),
                warnings=warnings,
                command_family=command_family,
                command_permission_class="dangerous",
            )

        return ToolValidationResult(
            allowed=True,
            warnings=warnings,
            command_family=command_family,
        )


class CommandSafetyValidator(ToolValidator):
    """Classifies shell commands into risk families and blocks dangerous patterns."""

    _BLOCKED_PATTERNS = [
        "rm -rf /",
        "> /dev/sda",
        "dd if=",
        ":(){ :|:& };:",
        "mkfs.",
    ]

    _DESTRUCTIVE_PREFIXES = ["rm -rf", "rm -r", "rmdir", "del /", "rd /"]
    _PRIVILEGE_PATTERNS = ["sudo", "su -", "doas", "pkexec"]
    _NETWORK_TOOLS = ["curl", "wget", "nc", "netcat", "nmap", "ssh", "scp", "ftp", "telnet"]

    def validate(
        self,
        tool_name: str,
        args: Dict[str, Any],
        context: ToolValidationContext,
    ) -> Optional[ToolValidationResult]:
        if not _is_command_tool(tool_name):
            return None
        command = str(args.get("command", "")).strip()
        if not command:
            return ToolValidationResult(
                allowed=False,
                reason="missing command",
                warnings=[],
            )
        for pattern in self._BLOCKED_PATTERNS:
            if pattern in command:
                return ToolValidationResult(
                    allowed=False,
                    reason=f"blocked pattern: {pattern}",
                    warnings=[],
                    command_permission_class="dangerous",
                    command_family="filesystem_destructive",
                )

        assessment = assess_command(command)
        family = assessment.family
        if family == "filesystem_destructive":
            return ToolValidationResult(
                allowed=False,
                reason="recursive or direct filesystem destruction detected",
                warnings=[],
                command_permission_class="dangerous",
                command_family=family,
            )
        if family == "privilege_escalation":
            return ToolValidationResult(
                allowed=False,
                reason="privilege escalation commands are not allowed",
                warnings=[],
                command_permission_class="dangerous",
                command_family=family,
            )
        if family == "unsafe_shell":
            return ToolValidationResult(
                allowed=False,
                reason="unsafe shell indirection detected",
                warnings=assessment.warnings,
                command_permission_class="dangerous",
                command_family=family,
            )
        return ToolValidationResult(
            allowed=True,
            warnings=assessment.warnings,
            command_permission_class=assessment.permission_class,
            command_family=family,
        )


class FilesystemPathValidator(ToolValidator):
    """Ensures filesystem paths stay within allowed roots."""

    def validate(
        self,
        tool_name: str,
        args: Dict[str, Any],
        context: ToolValidationContext,
    ) -> Optional[ToolValidationResult]:
        spec = _resolve_filesystem_spec(tool_name, args)
        if spec is None:
            return None
        target_path = spec.target_path
        if not target_path:
            return ToolValidationResult(
                allowed=False,
                reason="missing path",
                warnings=[],
            )
        if not context.roots:
            return ToolValidationResult(
                allowed=True,
                warnings=[],
                resolved_path=target_path,
            )
        try:
            requested = Path(target_path).expanduser().resolve()
            resolved = _resolve_path_under_policy(
                str(requested),
                mode=spec.mode,
                roots=context.roots,
                allow_any_read=(spec.mode == "read" and context.allow_any_read),
            )
            return ToolValidationResult(
                allowed=True,
                warnings=[],
                resolved_path=resolved,
            )
        except (ValueError, PermissionError) as exc:
            return ToolValidationResult(
                allowed=False,
                reason=str(exc),
                warnings=[],
            )


class FilesystemWatchlistValidator(ToolValidator):
    """Blocks writes to sensitive files (.env, ssh keys, etc.)."""

    _WATCHED = {
        ".env": "environment files",
        ".env.local": "environment files",
        ".env.production": "environment files",
        ".ssh/": "SSH directory",
        "id_rsa": "SSH private key",
        "id_ed25519": "SSH private key",
        ".aws/": "AWS credentials directory",
        "credentials": "credentials file",
        ".pgpass": "PostgreSQL password file",
        ".netrc": "netrc credentials file",
    }

    def validate(
        self,
        tool_name: str,
        args: Dict[str, Any],
        context: ToolValidationContext,
    ) -> Optional[ToolValidationResult]:
        spec = _resolve_filesystem_spec(tool_name, args)
        if spec is None or spec.mode != "write":
            return None
        target_path = spec.target_path
        if not target_path:
            return ToolValidationResult(
                allowed=False,
                reason="missing path",
                warnings=[],
            )
        normalized = str(target_path).replace("\\", "/").lower()
        for watch, reason in self._WATCHED.items():
            if watch.lower() in normalized or normalized.endswith(watch.lower()):
                return ToolValidationResult(
                    allowed=False,
                    reason=f"blocked by filesystem watchlist: {reason}",
                    warnings=[],
                )
        return None


class ContentSizeValidator(ToolValidator):
    """Enforces max payload size on writes."""

    def __init__(self, max_write_bytes: Optional[int] = None) -> None:
        self.max_write_bytes = max_write_bytes

    def validate(
        self,
        tool_name: str,
        args: Dict[str, Any],
        context: ToolValidationContext,
    ) -> Optional[ToolValidationResult]:
        spec = _resolve_filesystem_spec(tool_name, args)
        if spec is None or spec.mode != "write":
            return None
        limit = context.max_write_bytes or self.max_write_bytes
        if limit is None:
            return None
        content = str(args.get("content", ""))
        size = len(content.encode("utf-8"))
        if size > limit:
            return ToolValidationResult(
                allowed=False,
                reason=f"payload too large ({size} > {limit})",
                warnings=[],
            )
        return None


@dataclasses.dataclass
class _FilesystemSpec:
    mode: str  # "read" or "write"
    target_path: Optional[str]


def _resolve_filesystem_spec(
    tool_name: str, args: Dict[str, Any]
) -> Optional[_FilesystemSpec]:
    if tool_name == "fs_read_file":
        return _FilesystemSpec(
            mode="read",
            target_path=_extract_path_arg(args, ["file_path", "path", "filePath"]),
        )
    if tool_name == "fs_write_file":
        return _FilesystemSpec(
            mode="write",
            target_path=_extract_path_arg(args, ["file_path", "path", "filePath"]),
        )
    if tool_name == "fs_list_dir":
        return _FilesystemSpec(
            mode="read",
            target_path=_extract_path_arg(args, ["dir_path", "path", "dirPath"]),
        )
    return None


def _extract_path_arg(args: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for key in keys:
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _is_command_tool(tool_name: str) -> bool:
    return tool_name == "bash_run_command"


_READ_ONLY_COMMANDS = {
    "cat",
    "diff",
    "env",
    "file",
    "find",
    "git",
    "grep",
    "head",
    "help",
    "ls",
    "pydoc",
    "pwd",
    "rg",
    "sed",
    "sort",
    "stat",
    "tail",
    "tree",
    "uname",
    "wc",
    "which",
}

_BOUNDED_WRITE_COMMANDS = {
    "black",
    "cp",
    "mkdir",
    "mv",
    "pip",
    "pytest",
    "python",
    "python3",
    "ruff",
    "touch",
    "uv",
}

_GIT_READONLY_SUBCOMMANDS = {
    "status",
    "diff",
    "log",
    "show",
    "rev-parse",
    "blame",
    "grep",
    "ls-files",
    "branch",
}

_GIT_NETWORK_SUBCOMMANDS = {"fetch", "pull", "push", "clone"}
_GIT_DANGEROUS_SUBCOMMANDS = {"reset", "clean"}


def assess_command(command: str) -> CommandAssessment:
    """Classify a shell command into a family and permission class."""
    tokens = _tokenize_command(command)
    lowered = command.lower()
    if not tokens:
        return CommandAssessment(
            family="safe",
            permission_class="read_only",
        )

    first = tokens[0].lower()
    subcommand = _first_nonflag(tokens[1:])
    warnings: List[str] = []

    for prefix in CommandSafetyValidator._DESTRUCTIVE_PREFIXES:
        if lowered.startswith(prefix) or (" " + prefix) in lowered:
            return CommandAssessment(
                family="filesystem_destructive",
                permission_class="dangerous",
                executable=first,
                subcommand=subcommand,
            )

    if first in {"mkfs", "fdisk", "parted", "shred"}:
        return CommandAssessment(
            family="filesystem_destructive",
            permission_class="dangerous",
            executable=first,
            subcommand=subcommand,
        )

    for pat in CommandSafetyValidator._PRIVILEGE_PATTERNS:
        if pat in lowered:
            return CommandAssessment(
                family="privilege_escalation",
                permission_class="dangerous",
                executable=first,
                subcommand=subcommand,
            )

    if "()" in command and "&" in command:
        return CommandAssessment(
            family="unsafe_shell",
            permission_class="dangerous",
            executable=first,
            subcommand=subcommand,
            warnings=["fork-bomb-like shell structure detected"],
        )

    if first in {"sh", "bash", "zsh", "eval"} and any(
        part in command for part in (" -c ", "$(", "`")
    ):
        return CommandAssessment(
            family="unsafe_shell",
            permission_class="dangerous",
            executable=first,
            subcommand=subcommand,
            warnings=["nested shell indirection detected"],
        )

    if first in CommandSafetyValidator._NETWORK_TOOLS:
        return CommandAssessment(
            family="network",
            permission_class="network",
            executable=first,
            subcommand=subcommand,
        )

    if first == "git":
        return _assess_git_command(tokens, lowered)

    if first in _READ_ONLY_COMMANDS:
        return CommandAssessment(
            family="safe",
            permission_class="read_only",
            executable=first,
            subcommand=subcommand,
        )

    if first in _BOUNDED_WRITE_COMMANDS:
        return CommandAssessment(
            family="safe",
            permission_class="bounded_write",
            executable=first,
            subcommand=subcommand,
        )

    if re.search(r"[<>]", command):
        warnings.append("shell redirection detected")
        return CommandAssessment(
            family="safe",
            permission_class="bounded_write",
            executable=first,
            subcommand=subcommand,
            warnings=warnings,
        )

    return CommandAssessment(
        family="safe",
        permission_class="bounded_write",
        executable=first,
        subcommand=subcommand,
        warnings=["unclassified command treated as bounded_write"],
    )


def _assess_git_command(tokens: List[str], lowered: str) -> CommandAssessment:
    subcommand = _first_nonflag(tokens[1:])
    if subcommand in _GIT_NETWORK_SUBCOMMANDS:
        return CommandAssessment(
            family="network",
            permission_class="network",
            executable="git",
            subcommand=subcommand,
        )
    if subcommand in _GIT_DANGEROUS_SUBCOMMANDS:
        if "--hard" in tokens or "-fd" in tokens or "-xdf" in tokens or "-xffd" in tokens:
            return CommandAssessment(
                family="filesystem_destructive",
                permission_class="dangerous",
                executable="git",
                subcommand=subcommand,
            )
        return CommandAssessment(
            family="safe",
            permission_class="bounded_write",
            executable="git",
            subcommand=subcommand,
            warnings=["git history or worktree rewrite command detected"],
        )
    if subcommand in _GIT_READONLY_SUBCOMMANDS or subcommand is None:
        return CommandAssessment(
            family="safe",
            permission_class="read_only",
            executable="git",
            subcommand=subcommand,
        )
    return CommandAssessment(
        family="safe",
        permission_class="bounded_write",
        executable="git",
        subcommand=subcommand,
    )


def _tokenize_command(command: str) -> List[str]:
    import shlex

    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def _first_nonflag(tokens: List[str]) -> Optional[str]:
    for token in tokens:
        if token and not token.startswith("-"):
            return token.lower()
    return None


def _resolve_path_under_policy(
    target_path: str,
    mode: str,
    roots: List[str],
    allow_any_read: bool = False,
) -> str:
    target = Path(target_path).expanduser().resolve()
    if not roots:
        return str(target)
    if allow_any_read and mode == "read":
        return str(target)
    for root in roots:
        root_path = Path(root).expanduser().resolve()
        try:
            target.relative_to(root_path)
            return str(target)
        except ValueError:
            continue
    raise PermissionError(
        f"Path {target} is outside allowed roots: {[str(r) for r in roots]}"
    )
