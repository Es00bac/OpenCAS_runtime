"""Stage-aware generation policy for creativity, reasoning, and reflection."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class GenerationPhase(str, Enum):
    CONVERSATION = "conversation"
    BRAINSTORM = "brainstorm"
    DIVERGE = "diverge"
    OUTLINE = "outline"
    DRAFT = "draft"
    REVISE = "revise"
    VERIFY = "verify"
    EXECUTE = "execute"
    DAYDREAM = "daydream"
    DREAM = "dream"
    SYNTHESIZE = "synthesize"
    REPAIR = "repair"


class GenerationDomain(str, Enum):
    GENERAL = "general"
    CREATIVE_WRITING = "creative_writing"
    CODING = "coding"
    RESEARCH = "research"
    AUTONOMY = "autonomy"
    SAFETY = "safety"


class SomaticGenerationInfluence(BaseModel):
    enabled: bool = True
    max_temperature_delta: float = 0.15
    fatigue_clamp: bool = True
    tension_clamp: bool = True

    @field_validator("max_temperature_delta", mode="before")
    @classmethod
    def _clamp_delta(cls, value: Any) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = 0.15
        return max(0.0, min(0.4, parsed))


class GenerationProfile(BaseModel):
    primary_knob: str = "temperature"
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    top_h: Optional[float] = None
    candidate_count: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    reasoning_effort: Optional[str] = None
    max_output_tokens: Optional[int] = None
    summary: str = ""
    authority: str = "normal"

    @field_validator("primary_knob", mode="before")
    @classmethod
    def _normalize_primary_knob(cls, value: Any) -> str:
        cleaned = str(value or "temperature").strip().lower()
        return cleaned if cleaned in {"temperature", "top_p", "top_k"} else "temperature"


class LearnedGenerationPresetSettings(BaseModel):
    enabled: bool = True
    min_evidence_count: int = 5
    min_success_rate: float = 0.65
    max_active_presets: int = 64

    @field_validator("min_evidence_count", "max_active_presets", mode="before")
    @classmethod
    def _clamp_counts(cls, value: Any) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = 0
        return max(0, parsed)

    @field_validator("min_success_rate", mode="before")
    @classmethod
    def _clamp_rate(cls, value: Any) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = 0.65
        return max(0.0, min(1.0, parsed))


class LearnedGenerationPreset(BaseModel):
    preset_id: str
    work_type: str
    phase: GenerationPhase | str
    domain: GenerationDomain | str = GenerationDomain.GENERAL
    profile: GenerationProfile
    source: str = "unknown"
    evidence_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    last_used_at: Optional[str] = None
    disabled: bool = False
    notes: list[str] = Field(default_factory=list)

    @field_validator("preset_id", "work_type", mode="before")
    @classmethod
    def _require_text(cls, value: Any) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            raise ValueError("learned preset id and work_type are required")
        return cleaned

    @field_validator("phase", mode="before")
    @classmethod
    def _normalize_phase(cls, value: Any) -> str:
        return GenerationPhase(_enum_text(value).strip().lower()).value

    @field_validator("domain", mode="before")
    @classmethod
    def _normalize_domain(cls, value: Any) -> str:
        return GenerationDomain(_enum_text(value or "general").strip().lower()).value

    @property
    def success_rate(self) -> float:
        evidence = max(0, int(self.evidence_count))
        if evidence <= 0:
            return 0.0
        return max(0, int(self.success_count)) / evidence

    def is_mature(self, settings: LearnedGenerationPresetSettings) -> bool:
        if self.disabled:
            return False
        return (
            int(self.evidence_count) >= int(settings.min_evidence_count)
            and self.success_rate >= float(settings.min_success_rate)
        )


def default_generation_profiles() -> Dict[str, GenerationProfile]:
    return {
        "conversation": GenerationProfile(
            temperature=0.5,
            summary="Normal conversation uses somatic temperature unless a phase requests more structure.",
        ),
        "brainstorm": GenerationProfile(
            temperature=0.9,
            candidate_count=3,
            summary="Wide option generation with later filtering.",
        ),
        "diverge": GenerationProfile(
            temperature=1.0,
            candidate_count=3,
            min_p=0.08,
            top_h=0.45,
            summary="Low-risk surprising associations for creative exploration.",
        ),
        "outline": GenerationProfile(
            temperature=0.62,
            reasoning_effort="high",
            summary="Structure and continuity before prose.",
        ),
        "draft": GenerationProfile(
            temperature=0.82,
            summary="Expressive prose constrained by supplied state.",
        ),
        "revise": GenerationProfile(
            temperature=0.4,
            summary="Tighten coherence, style, and consistency.",
        ),
        "verify": GenerationProfile(
            temperature=0.0,
            summary="Deterministic checking and structured review.",
        ),
        "execute": GenerationProfile(
            temperature=0.2,
            summary="Tool, coding, and action steps stay conservative.",
        ),
        "daydream": GenerationProfile(
            temperature=0.78,
            top_p=0.9,
            top_k=70,
            summary="Reflective sparks without execution authority.",
            authority="reflective_proposal_only",
        ),
        "dream": GenerationProfile(
            temperature=0.95,
            top_p=0.92,
            top_k=85,
            summary="Loose private association without execution authority.",
            authority="reflective_proposal_only",
        ),
        "synthesize": GenerationProfile(
            temperature=0.5,
            summary="Merge evidence and notes into a useful shape.",
        ),
        "repair": GenerationProfile(
            temperature=0.32,
            reasoning_effort="high",
            summary="Diagnose and recover after failure without novelty churn.",
        ),
    }


class GenerationPolicyConfig(BaseModel):
    enabled: bool = True
    default_profile: str = "balanced"
    preserve_explicit_payload_overrides: bool = True
    provider_strategy: str = "capability_filtered"
    somatic_influence: SomaticGenerationInfluence = Field(
        default_factory=SomaticGenerationInfluence
    )
    profiles: Dict[str, GenerationProfile] = Field(
        default_factory=default_generation_profiles
    )
    local_sampler_experiments: Dict[str, bool] = Field(
        default_factory=lambda: {"min_p": False, "top_h": False, "mirostat": False}
    )
    learned_presets: LearnedGenerationPresetSettings = Field(
        default_factory=LearnedGenerationPresetSettings
    )
    learned_preset_records: Dict[str, LearnedGenerationPreset] = Field(
        default_factory=dict
    )

    def normalized(self) -> "GenerationPolicyConfig":
        profiles = default_generation_profiles()
        profiles.update(self.profiles or {})
        experiments = {"min_p": False, "top_h": False, "mirostat": False}
        experiments.update(self.local_sampler_experiments or {})
        learned_records = dict(self.learned_preset_records or {})
        if self.learned_presets.max_active_presets:
            learned_records = dict(
                list(learned_records.items())[: self.learned_presets.max_active_presets]
            )
        return self.model_copy(
            update={
                "profiles": profiles,
                "local_sampler_experiments": experiments,
                "learned_preset_records": learned_records,
            }
        )


@dataclass
class GenerationPolicyRequest:
    phase: GenerationPhase | str
    domain: GenerationDomain | str = GenerationDomain.GENERAL
    risk_level: str = "normal"
    continuity_pressure: float = 0.0
    novelty_pressure: float = 0.0
    structured_output: bool = False
    long_form: bool = False
    source: str = "unknown"
    somatic: Any = None
    work_type: str | None = None
    memory_focus: list[str] = field(default_factory=list)
    variation_seed: str | None = None

    @classmethod
    def from_any(cls, value: "GenerationPolicyRequest | Mapping[str, Any]") -> "GenerationPolicyRequest":
        if isinstance(value, cls):
            return value
        return cls(**dict(value))

    def normalized(self) -> "GenerationPolicyRequest":
        return GenerationPolicyRequest(
            phase=GenerationPhase(_enum_text(self.phase).lower()),
            domain=GenerationDomain(_enum_text(self.domain).lower()),
            risk_level=str(self.risk_level or "normal"),
            continuity_pressure=_clamp01(self.continuity_pressure),
            novelty_pressure=_clamp01(self.novelty_pressure),
            structured_output=bool(self.structured_output),
            long_form=bool(self.long_form),
            source=str(self.source or "unknown"),
            somatic=self.somatic,
            work_type=str(self.work_type).strip() if self.work_type else None,
            memory_focus=[str(item).strip() for item in self.memory_focus if str(item).strip()],
            variation_seed=str(self.variation_seed).strip() if self.variation_seed else None,
        )


@dataclass
class GenerationSamplingPolicy:
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    top_h: float | None = None
    candidate_count: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    reasoning_effort: str | None = None
    max_output_tokens: int | None = None
    variation_seed: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class EffectiveGenerationPolicy:
    requested: GenerationSamplingPolicy
    effective: Dict[str, Any]
    unsupported: Dict[str, Any]
    profile_name: str
    provider_id: str | None
    model_ref: str | None
    telemetry: Dict[str, Any]

    def trace_fields(self) -> Dict[str, Any]:
        fields = {
            "generation_phase": self.telemetry.get("phase"),
            "generation_domain": self.telemetry.get("domain"),
            "generation_profile": self.profile_name,
            "generation_authority": self.telemetry.get("authority"),
            "generation_requested_sampling": self.telemetry.get("requested_sampling"),
            "generation_effective_sampling": self.effective,
            "generation_unsupported_sampling": self.unsupported,
            "generation_adjustments": self.telemetry.get("adjustments"),
            "generation_somatic_delta": self.telemetry.get("somatic_delta"),
            "generation_explicit_overrides": self.telemetry.get("explicit_overrides"),
            "generation_memory_focus": self.telemetry.get("memory_focus"),
            "sampling_variation_seed": self.telemetry.get("sampling_variation_seed"),
        }
        return {key: value for key, value in fields.items() if value not in (None, [], {})}


_SAMPLING_KEYS = {
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "top_h",
    "candidate_count",
    "presence_penalty",
    "frequency_penalty",
    "max_output_tokens",
}

_HOSTED_TEMPERATURE_ONLY_PROVIDERS = {
    "anthropic",
    "codex-cli",
    "google",
    "gemini",
    "kimi-coding",
    "openai",
    "openai-codex",
    "zai",
    "zai-anthropic",
    "zai-coding",
}

_LOCAL_PROVIDER_PREFIXES = ("local", "llama", "ollama", "vllm", "lmstudio")

_PHASE_RANGES = {
    GenerationPhase.VERIFY: (0.0, 0.2),
    GenerationPhase.EXECUTE: (0.1, 0.35),
    GenerationPhase.REPAIR: (0.2, 0.45),
    GenerationPhase.BRAINSTORM: (0.75, 1.1),
    GenerationPhase.DIVERGE: (0.85, 1.2),
    GenerationPhase.OUTLINE: (0.45, 0.75),
    GenerationPhase.DRAFT: (0.55, 1.0),
    GenerationPhase.REVISE: (0.25, 0.6),
    GenerationPhase.SYNTHESIZE: (0.35, 0.7),
    GenerationPhase.DAYDREAM: (0.6, 0.95),
    GenerationPhase.DREAM: (0.7, 1.15),
    GenerationPhase.CONVERSATION: (0.25, 0.8),
}


class GenerationPolicyResolver:
    """Resolve phase, mood, risk, and provider support into sampling payloads."""

    def __init__(self, config: GenerationPolicyConfig | None = None) -> None:
        self.config = (config or GenerationPolicyConfig()).normalized()

    def set_config(self, config: GenerationPolicyConfig) -> None:
        self.config = config.normalized()

    def resolve(
        self,
        request: GenerationPolicyRequest | Mapping[str, Any],
        *,
        provider_id: str | None,
        model_ref: str | None,
        explicit_payload: Mapping[str, Any] | None = None,
    ) -> EffectiveGenerationPolicy:
        req = GenerationPolicyRequest.from_any(request).normalized()
        profile_name = req.phase.value
        learned = self._select_learned_preset(req)
        if learned is not None:
            profile_name = f"learned:{learned.preset_id}"
            profile = learned.profile
        else:
            profile = self.config.profiles.get(profile_name) or self.config.profiles["conversation"]
        requested = self._requested_policy(req, profile)
        effective = self._policy_to_payload(requested, profile, provider_id)
        unsupported = self._unsupported_policy(requested, effective)
        explicit_overrides = []
        if self.config.preserve_explicit_payload_overrides and explicit_payload:
            for key in sorted(_SAMPLING_KEYS | {"reasoning_effort"}):
                if key in explicit_payload:
                    effective[key] = explicit_payload[key]
                    unsupported.pop(key, None)
                    explicit_overrides.append(key)
        telemetry = {
            "phase": req.phase.value,
            "domain": req.domain.value,
            "source": req.source,
            "risk_level": req.risk_level,
            "authority": profile.authority,
            "requested_sampling": {
                key: value
                for key, value in requested.__dict__.items()
                if value not in (None, [], {})
            },
            "adjustments": list(requested.notes),
            "explicit_overrides": explicit_overrides,
            "memory_focus": self._memory_focus(req),
        }
        if requested.variation_seed:
            telemetry["sampling_variation_seed"] = requested.variation_seed
        if learned is not None:
            telemetry.update(
                {
                    "learned_preset_id": learned.preset_id,
                    "learned_preset_work_type": learned.work_type,
                    "learned_preset_success_rate": learned.success_rate,
                    "learned_preset_evidence_count": learned.evidence_count,
                    "learned_preset_source": learned.source,
                }
            )
        if "__somatic_delta" in effective:
            telemetry["somatic_delta"] = effective.pop("__somatic_delta")
        if not self.config.enabled:
            requested.notes.append("generation_policy_disabled")
            effective = {}
            unsupported = {}
            telemetry["adjustments"] = list(requested.notes)
        return EffectiveGenerationPolicy(
            requested=requested,
            effective={key: value for key, value in effective.items() if value is not None},
            unsupported=unsupported,
            profile_name=profile_name,
            provider_id=provider_id,
            model_ref=model_ref,
            telemetry=telemetry,
        )

    @staticmethod
    def _memory_focus(req: GenerationPolicyRequest) -> list[str]:
        focus = list(req.memory_focus)
        if req.domain == GenerationDomain.CREATIVE_WRITING and req.long_form:
            focus.extend(
                [
                    "project_memory",
                    "autobiographical_memory",
                    "affective_writing_context",
                    "prior_ideas",
                    "discarded_ideas",
                    "current_file_evidence",
                ]
            )
        elif req.phase in {GenerationPhase.DAYDREAM, GenerationPhase.DREAM}:
            focus.extend(["reflective_memory", "association_memory", "active_work_context"])
        elif req.phase in {GenerationPhase.EXECUTE, GenerationPhase.REPAIR}:
            focus.extend(["tool_memory", "skill_memory", "project_memory", "current_file_evidence"])
        if req.work_type:
            focus.append(f"work_type:{req.work_type}")
        return list(dict.fromkeys(item for item in focus if item))

    def _select_learned_preset(
        self,
        req: GenerationPolicyRequest,
    ) -> LearnedGenerationPreset | None:
        if not self.config.learned_presets.enabled or not req.work_type:
            return None
        wanted = req.work_type.strip().casefold()
        matches = []
        for preset in self.config.learned_preset_records.values():
            if preset.work_type.strip().casefold() != wanted:
                continue
            preset_phase = (
                preset.phase.value if isinstance(preset.phase, GenerationPhase) else str(preset.phase)
            )
            preset_domain = (
                preset.domain.value
                if isinstance(preset.domain, GenerationDomain)
                else str(preset.domain)
            )
            if preset_phase != req.phase.value:
                continue
            if preset_domain not in {req.domain.value, GenerationDomain.GENERAL.value}:
                continue
            if not preset.is_mature(self.config.learned_presets):
                continue
            matches.append(preset)
        if not matches:
            return None
        matches.sort(
            key=lambda item: (item.success_rate, int(item.evidence_count), item.preset_id),
            reverse=True,
        )
        return matches[0]

    def phase_summary(self) -> Dict[str, Dict[str, Any]]:
        summary: Dict[str, Dict[str, Any]] = {}
        for phase in GenerationPhase:
            policy = self.resolve(
                GenerationPolicyRequest(phase=phase),
                provider_id="openai-codex",
                model_ref=None,
            )
            summary[phase.value] = {
                "temperature": policy.effective.get("temperature"),
                "authority": policy.telemetry.get("authority"),
                "summary": self.config.profiles[phase.value].summary,
                "novelty": _phase_nominal_novelty(phase),
            }
        return summary

    def _requested_policy(
        self,
        req: GenerationPolicyRequest,
        profile: GenerationProfile,
    ) -> GenerationSamplingPolicy:
        notes: list[str] = []
        temperature = profile.temperature
        if temperature is None:
            temperature = 0.5
        if req.novelty_pressure > 0.0 and req.phase in {
            GenerationPhase.BRAINSTORM,
            GenerationPhase.DIVERGE,
            GenerationPhase.DRAFT,
            GenerationPhase.DAYDREAM,
            GenerationPhase.DREAM,
        }:
            temperature += req.novelty_pressure * 0.12
            notes.append("novelty_pressure")
        if req.continuity_pressure > 0.0:
            temperature -= req.continuity_pressure * (0.2 if req.long_form else 0.12)
            notes.append("continuity_pressure")
        somatic_delta = 0.0
        if self.config.somatic_influence.enabled and req.somatic is not None:
            somatic_delta = self._somatic_temperature_delta(req)
            temperature += somatic_delta
            notes.append("somatic_delta")
        if req.structured_output:
            temperature = min(temperature, 0.2)
            notes.append("structured_output_clamp")
        if req.phase in {GenerationPhase.VERIFY, GenerationPhase.EXECUTE, GenerationPhase.REPAIR}:
            tension = _somatic_value(req.somatic, "tension")
            if self.config.somatic_influence.tension_clamp and tension > 0.5:
                temperature = min(temperature, 0.3)
                notes.append("tension_clamp")
        lo, hi = _PHASE_RANGES[req.phase]
        temperature = round(max(lo, min(hi, temperature)), 3)
        top_p = profile.top_p
        top_k = profile.top_k
        variation_seed = None
        if req.phase in {GenerationPhase.DAYDREAM, GenerationPhase.DREAM}:
            temperature, top_p, top_k, variation_seed = self._reflective_sampler_variation(
                req=req,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            notes.append("bounded_sampler_variation")
        policy = GenerationSamplingPolicy(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=profile.min_p,
            top_h=profile.top_h,
            candidate_count=profile.candidate_count,
            presence_penalty=profile.presence_penalty,
            frequency_penalty=profile.frequency_penalty,
            reasoning_effort=profile.reasoning_effort,
            max_output_tokens=profile.max_output_tokens,
            variation_seed=variation_seed,
            notes=notes,
        )
        policy.__dict__["__somatic_delta"] = round(somatic_delta, 3)
        return policy

    @staticmethod
    def _reflective_sampler_variation(
        *,
        req: GenerationPolicyRequest,
        temperature: float,
        top_p: float | None,
        top_k: int | None,
    ) -> tuple[float, float, int, str]:
        seed = req.variation_seed or f"{req.phase.value}:{uuid4().hex}"
        rng = random.Random(seed)
        top_p_base = 0.9 if top_p is None else float(top_p)
        top_k_base = 70 if top_k is None else int(top_k)
        varied_temperature = temperature + rng.uniform(-0.08, 0.08)
        varied_top_p = top_p_base + rng.uniform(-0.08, 0.08)
        varied_top_k = top_k_base + rng.choice([-30, -20, -10, 0, 10, 20, 30, 40])
        lo, hi = _PHASE_RANGES[req.phase]
        return (
            round(max(lo, min(hi, varied_temperature)), 3),
            round(max(0.78, min(0.98, varied_top_p)), 3),
            int(max(30, min(110, varied_top_k))),
            seed,
        )

    def _somatic_temperature_delta(self, req: GenerationPolicyRequest) -> float:
        influence = self.config.somatic_influence
        arousal = _somatic_value(req.somatic, "arousal")
        fatigue = _somatic_value(req.somatic, "fatigue")
        focus = _somatic_value(req.somatic, "focus")
        tension = _somatic_value(req.somatic, "tension")
        certainty = _somatic_value(req.somatic, "certainty", default=0.5)
        delta = (arousal * 0.10) - (fatigue * 0.12)
        if req.phase in {GenerationPhase.EXECUTE, GenerationPhase.REPAIR, GenerationPhase.VERIFY}:
            delta -= focus * 0.08
        else:
            delta -= focus * 0.03
        if influence.tension_clamp and tension > 0.5:
            delta -= tension * 0.08
        if certainty < 0.35 and req.phase not in {GenerationPhase.DAYDREAM, GenerationPhase.DREAM}:
            delta -= 0.04
        return round(max(-influence.max_temperature_delta, min(influence.max_temperature_delta, delta)), 3)

    def _policy_to_payload(
        self,
        requested: GenerationSamplingPolicy,
        profile: GenerationProfile,
        provider_id: str | None,
    ) -> Dict[str, Any]:
        provider = str(provider_id or "").strip().lower()
        hosted_temperature_only = provider in _HOSTED_TEMPERATURE_ONLY_PROVIDERS
        local_provider = provider.startswith(_LOCAL_PROVIDER_PREFIXES)
        payload: Dict[str, Any] = {}
        if profile.primary_knob == "top_p" and requested.top_p is not None:
            payload["top_p"] = requested.top_p
        elif requested.temperature is not None:
            payload["temperature"] = requested.temperature
        if not hosted_temperature_only:
            if profile.primary_knob != "top_p" and requested.top_p is not None:
                payload["top_p"] = requested.top_p
            if requested.top_k is not None:
                payload["top_k"] = requested.top_k
        if local_provider:
            experiments = self.config.local_sampler_experiments
            if experiments.get("min_p") and requested.min_p is not None:
                payload["min_p"] = requested.min_p
            if experiments.get("top_h") and requested.top_h is not None:
                payload["top_h"] = requested.top_h
        if (provider.startswith("google") or provider.startswith("gemini")) and requested.candidate_count is not None:
            payload["candidate_count"] = requested.candidate_count
        for key in ("presence_penalty", "frequency_penalty", "reasoning_effort", "max_output_tokens"):
            value = getattr(requested, key)
            if value is not None:
                payload[key] = value
        payload["__somatic_delta"] = requested.__dict__.get("__somatic_delta", 0.0)
        return payload

    @staticmethod
    def _unsupported_policy(
        requested: GenerationSamplingPolicy,
        effective: Mapping[str, Any],
    ) -> Dict[str, Any]:
        unsupported: Dict[str, Any] = {}
        for key in _SAMPLING_KEYS:
            value = getattr(requested, key)
            if value is not None and key not in effective:
                unsupported[key] = value
        return unsupported


def generation_policy_settings_path(state_dir: Path) -> Path:
    return Path(state_dir).expanduser() / "runtime_generation_policy.json"


def load_persisted_generation_policy(state_dir: Path) -> Optional[GenerationPolicyConfig]:
    path = generation_policy_settings_path(state_dir)
    if not path.exists():
        return None
    try:
        return GenerationPolicyConfig.model_validate_json(path.read_text(encoding="utf-8")).normalized()
    except Exception:
        return None


def save_persisted_generation_policy(
    state_dir: Path,
    config: GenerationPolicyConfig,
) -> Path:
    path = generation_policy_settings_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.normalized().model_dump(mode="json"), indent=2), encoding="utf-8")
    return path


def upsert_learned_generation_preset(
    config: GenerationPolicyConfig,
    preset: LearnedGenerationPreset,
) -> GenerationPolicyConfig:
    normalized = config.normalized()
    records = dict(normalized.learned_preset_records)
    records[preset.preset_id] = preset
    return normalized.model_copy(update={"learned_preset_records": records}).normalized()


def _somatic_value(source: Any, key: str, *, default: float = 0.0) -> float:
    if source is None:
        return default
    if isinstance(source, Mapping):
        value = source.get(key, default)
    else:
        value = getattr(source, key, default)
    return _clamp01(value)


def _clamp01(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = 0.0
    return max(0.0, min(1.0, parsed))


def _enum_text(value: Any) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def _phase_nominal_novelty(phase: GenerationPhase) -> float:
    return {
        GenerationPhase.VERIFY: 0.0,
        GenerationPhase.EXECUTE: 0.1,
        GenerationPhase.REPAIR: 0.2,
        GenerationPhase.REVISE: 0.25,
        GenerationPhase.SYNTHESIZE: 0.35,
        GenerationPhase.OUTLINE: 0.45,
        GenerationPhase.CONVERSATION: 0.45,
        GenerationPhase.DRAFT: 0.6,
        GenerationPhase.DAYDREAM: 0.65,
        GenerationPhase.BRAINSTORM: 0.8,
        GenerationPhase.DIVERGE: 0.9,
        GenerationPhase.DREAM: 0.95,
    }[phase]
