"""Plugin bundle helpers for archive-based phase-two SDK workflows."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Tuple

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from .manifest import normalize_plugin_manifest, validate_plugin_manifest, load_plugin_manifest

PLUGIN_BUNDLE_SUFFIX = ".opencas-plugin.zip"
_SUPPORTED_SIGNATURE_ALGORITHMS = {"ed25519"}


class PluginPackageError(ValueError):
    """Raised when plugin bundle creation or extraction fails."""


def _compute_bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _compute_bundle_sha256(bundle_path: Path) -> str:
    hasher = hashlib.sha256()
    with bundle_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _safe_bundle_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members: list[zipfile.ZipInfo] = []
    for info in archive.infolist():
        member_path = PurePosixPath(info.filename)
        if not info.filename or member_path.is_absolute():
            raise PluginPackageError(f"bundle member has invalid path: {info.filename!r}")
        if any(part in {"", ".", ".."} for part in member_path.parts):
            raise PluginPackageError(f"bundle member escapes package root: {info.filename!r}")
        members.append(info)
    return members


def _normalize_public_key_text(value: str | bytes) -> str:
    key = _load_ed25519_public_key(value)
    raw = key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return base64.b64encode(raw).decode("ascii")


def _public_key_fingerprint(value: str | bytes) -> str:
    normalized = _normalize_public_key_text(value)
    return _compute_bytes_sha256(base64.b64decode(normalized.encode("ascii")))


def _load_ed25519_private_key(value: str | bytes) -> ed25519.Ed25519PrivateKey:
    raw = value.encode("utf-8") if isinstance(value, str) else bytes(value)
    stripped = raw.strip()
    try:
        if stripped.startswith(b"-----BEGIN"):
            key = serialization.load_pem_private_key(stripped, password=None)
            if not isinstance(key, ed25519.Ed25519PrivateKey):
                raise PluginPackageError("private key must be an Ed25519 key")
            return key
    except ValueError as exc:
        raise PluginPackageError(f"invalid Ed25519 private key: {exc}") from exc

    try:
        decoded = base64.b64decode(stripped, validate=True)
    except Exception as exc:
        raise PluginPackageError(f"invalid Ed25519 private key encoding: {exc}") from exc
    if len(decoded) != 32:
        raise PluginPackageError("Ed25519 private key must decode to 32 raw bytes")
    return ed25519.Ed25519PrivateKey.from_private_bytes(decoded)


def _load_ed25519_public_key(value: str | bytes) -> ed25519.Ed25519PublicKey:
    raw = value.encode("utf-8") if isinstance(value, str) else bytes(value)
    stripped = raw.strip()
    try:
        if stripped.startswith(b"-----BEGIN"):
            key = serialization.load_pem_public_key(stripped)
            if not isinstance(key, ed25519.Ed25519PublicKey):
                raise PluginPackageError("public key must be an Ed25519 key")
            return key
    except ValueError as exc:
        raise PluginPackageError(f"invalid Ed25519 public key: {exc}") from exc

    try:
        decoded = base64.b64decode(stripped, validate=True)
    except Exception as exc:
        raise PluginPackageError(f"invalid Ed25519 public key encoding: {exc}") from exc
    if len(decoded) != 32:
        raise PluginPackageError("Ed25519 public key must decode to 32 raw bytes")
    return ed25519.Ed25519PublicKey.from_public_bytes(decoded)


def _manifest_without_signatures(manifest: Dict[str, Any]) -> Dict[str, Any]:
    normalized = normalize_plugin_manifest(manifest)
    payload_manifest = json.loads(json.dumps(normalized))
    distribution = payload_manifest.get("distribution")
    if isinstance(distribution, dict):
        distribution.pop("signatures", None)
    return payload_manifest


def _source_member_hashes(source_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for file_path in sorted(source_dir.rglob("*")):
        if not file_path.is_file():
            continue
        relative = file_path.relative_to(source_dir).as_posix()
        if relative == "plugin.json":
            continue
        hashes[relative] = _compute_bytes_sha256(file_path.read_bytes())
    return hashes


def _archive_member_hashes(archive: zipfile.ZipFile, members: list[zipfile.ZipInfo]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for info in members:
        if info.is_dir() or info.filename == "plugin.json":
            continue
        hashes[info.filename] = _compute_bytes_sha256(archive.read(info.filename))
    return hashes


def build_plugin_signature_payload(
    manifest: Dict[str, Any],
    member_hashes: Dict[str, str],
) -> bytes:
    payload = {
        "manifest": _manifest_without_signatures(manifest),
        "members": dict(sorted(member_hashes.items())),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_plugin_directory(
    source_dir: Path | str,
    *,
    key_id: str,
    private_key: str | bytes | Path,
    algorithm: str = "ed25519",
) -> Dict[str, Any]:
    """Sign a plugin directory by updating plugin.json with a signature entry."""

    if algorithm not in _SUPPORTED_SIGNATURE_ALGORITHMS:
        raise PluginPackageError(f"unsupported signature algorithm: {algorithm}")

    source_dir = Path(source_dir)
    manifest_path = source_dir / "plugin.json"
    manifest = load_plugin_manifest(manifest_path)
    member_hashes = _source_member_hashes(source_dir)
    payload = build_plugin_signature_payload(manifest, member_hashes)

    key_material: str | bytes
    if isinstance(private_key, Path):
        key_material = private_key.read_bytes()
    else:
        key_material = private_key
    private = _load_ed25519_private_key(key_material)
    public_key_raw = private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    signature = private.sign(payload)

    distribution = dict(manifest.get("distribution") or {})
    signatures = [
        dict(item)
        for item in (distribution.get("signatures") or [])
        if isinstance(item, dict)
        and str(item.get("key_id", "")).strip() != key_id
    ]
    signature_entry = {
        "key_id": key_id,
        "algorithm": algorithm,
        "public_key": base64.b64encode(public_key_raw).decode("ascii"),
        "signature": base64.b64encode(signature).decode("ascii"),
    }
    signatures.append(signature_entry)
    distribution["signatures"] = signatures
    manifest["distribution"] = distribution
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Re-validate after writing so signing never leaves the manifest stale.
    load_plugin_manifest(manifest_path)
    return {
        **signature_entry,
        "payload_sha256": _compute_bytes_sha256(payload),
        "public_key_fingerprint": _public_key_fingerprint(signature_entry["public_key"]),
    }


def _verify_bundle_signatures(
    manifest: Dict[str, Any],
    member_hashes: Dict[str, str],
) -> Dict[str, Any]:
    distribution = manifest.get("distribution") if isinstance(manifest.get("distribution"), dict) else {}
    signatures = distribution.get("signatures") if isinstance(distribution, dict) else None
    if not isinstance(signatures, list) or not signatures:
        return {
            "present": False,
            "count": 0,
            "verified_count": 0,
            "entries": [],
        }

    payload = build_plugin_signature_payload(manifest, member_hashes)
    payload_sha256 = _compute_bytes_sha256(payload)
    entries: list[dict[str, Any]] = []
    verified_count = 0
    for raw_entry in signatures:
        if not isinstance(raw_entry, dict):
            entries.append(
                {
                    "verified": False,
                    "error": "signature entry is not an object",
                }
            )
            continue
        key_id = str(raw_entry.get("key_id", "")).strip()
        algorithm = str(raw_entry.get("algorithm", "")).strip()
        public_key_value = raw_entry.get("public_key")
        signature_value = raw_entry.get("signature")
        entry_payload: dict[str, Any] = {
            "key_id": key_id,
            "algorithm": algorithm,
            "payload_sha256": payload_sha256,
            "verified": False,
            "public_key": public_key_value if isinstance(public_key_value, str) else None,
            "public_key_fingerprint": None,
        }
        try:
            if algorithm not in _SUPPORTED_SIGNATURE_ALGORITHMS:
                raise PluginPackageError(f"unsupported signature algorithm: {algorithm or '<missing>'}")
            if not isinstance(public_key_value, str) or not public_key_value.strip():
                raise PluginPackageError("signature entry is missing public_key")
            if not isinstance(signature_value, str) or not signature_value.strip():
                raise PluginPackageError("signature entry is missing signature")
            public_key = _load_ed25519_public_key(public_key_value)
            signature_bytes = base64.b64decode(signature_value.encode("ascii"), validate=True)
            public_key.verify(signature_bytes, payload)
            entry_payload["verified"] = True
            entry_payload["public_key_fingerprint"] = _public_key_fingerprint(public_key_value)
            verified_count += 1
        except (InvalidSignature, PluginPackageError, ValueError, TypeError, binascii.Error) as exc:
            if isinstance(exc, InvalidSignature):
                entry_payload["error"] = "signature verification failed"
            else:
                entry_payload["error"] = str(exc)
            if isinstance(public_key_value, str) and public_key_value.strip():
                try:
                    entry_payload["public_key_fingerprint"] = _public_key_fingerprint(public_key_value)
                except PluginPackageError:
                    entry_payload["public_key_fingerprint"] = None
        entries.append(entry_payload)

    return {
        "present": True,
        "count": len(entries),
        "verified_count": verified_count,
        "payload_sha256": payload_sha256,
        "entries": entries,
    }


def _bundle_metadata(
    bundle_path: Path,
    members: list[zipfile.ZipInfo],
    manifest: Dict[str, Any],
    member_hashes: Dict[str, str],
) -> dict[str, Any]:
    return {
        "filename": bundle_path.name,
        "sha256": _compute_bundle_sha256(bundle_path),
        "size_bytes": bundle_path.stat().st_size,
        "member_count": len(members),
        "signatures": _verify_bundle_signatures(manifest, member_hashes),
    }


def _read_bundle_manifest(bundle_path: Path) -> tuple[Dict[str, Any], dict[str, Any]]:
    if not bundle_path.exists() or not bundle_path.is_file():
        raise PluginPackageError(f"plugin bundle does not exist: {bundle_path}")

    with zipfile.ZipFile(bundle_path, "r") as archive:
        members = _safe_bundle_members(archive)
        try:
            raw = archive.read("plugin.json")
        except KeyError as exc:
            raise PluginPackageError("plugin bundle must contain plugin.json at the archive root") from exc
        member_hashes = _archive_member_hashes(archive, members)
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise PluginPackageError(f"failed to parse bundle manifest: {exc}") from exc
    if not isinstance(decoded, dict):
        raise PluginPackageError("bundle manifest root must be a JSON object")
    manifest = normalize_plugin_manifest(decoded)
    errors = validate_plugin_manifest(manifest)
    if errors:
        raise PluginPackageError("; ".join(errors))
    return manifest, _bundle_metadata(bundle_path, members, manifest, member_hashes)


def build_plugin_bundle(
    source_dir: Path | str,
    output_path: Path | str | None = None,
) -> Path:
    """Build a plugin bundle archive from *source_dir* and return the bundle path."""

    source_dir = Path(source_dir)
    manifest_path = source_dir / "plugin.json"
    if not source_dir.exists() or not source_dir.is_dir():
        raise PluginPackageError(f"plugin source directory does not exist: {source_dir}")

    manifest = load_plugin_manifest(manifest_path)
    if output_path is None:
        safe_version = str(manifest.get("version", "0.0.1")).replace("/", "-")
        output_path = source_dir.parent / f"{manifest['id']}-{safe_version}{PLUGIN_BUNDLE_SUFFIX}"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(source_dir.rglob("*")):
            if not file_path.is_file():
                continue
            archive.write(file_path, file_path.relative_to(source_dir).as_posix())

    return output_path


def inspect_plugin_bundle(bundle_path: Path | str) -> Dict[str, Any]:
    """Read and validate the bundle manifest without extracting the archive."""

    bundle_path = Path(bundle_path)
    manifest, _ = _read_bundle_manifest(bundle_path)
    return manifest


def inspect_plugin_bundle_details(bundle_path: Path | str) -> Dict[str, Any]:
    """Read and validate bundle manifest plus archive metadata for operator preview."""

    bundle_path = Path(bundle_path)
    manifest, bundle = _read_bundle_manifest(bundle_path)
    return {"manifest": manifest, "bundle": bundle}


def extract_plugin_bundle(
    bundle_path: Path | str,
    destination_root: Path | str,
) -> Tuple[Path, Dict[str, Any]]:
    """Extract *bundle_path* into *destination_root* and return the manifest path plus manifest."""

    bundle_path = Path(bundle_path)
    destination_root = Path(destination_root)
    if not bundle_path.exists() or not bundle_path.is_file():
        raise PluginPackageError(f"plugin bundle does not exist: {bundle_path}")
    destination_root.mkdir(parents=True, exist_ok=True)

    temp_dir = Path(
        tempfile.mkdtemp(prefix="plugin-bundle-", dir=str(destination_root))
    )
    try:
        with zipfile.ZipFile(bundle_path, "r") as archive:
            members = _safe_bundle_members(archive)
            archive.extractall(temp_dir, members)

        manifest_path = temp_dir / "plugin.json"
        if not manifest_path.exists():
            raise PluginPackageError("plugin bundle must contain plugin.json at the archive root")

        manifest = load_plugin_manifest(manifest_path)
        final_dir = destination_root / manifest["id"]
        if final_dir.exists():
            shutil.rmtree(final_dir)
        temp_dir.replace(final_dir)
        return final_dir / "plugin.json", manifest
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise
