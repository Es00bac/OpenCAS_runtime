"""Tests for Bulma → OpenCAS domain mappers."""

from datetime import datetime, timezone

from opencas.legacy.loader import load_json
from opencas.legacy.mapper import (
    bulma_episode_uuid,
    map_bulma_episode,
    map_bulma_memory_edge,
    map_bulma_history,
    map_bulma_identity,
    map_bulma_spark,
    map_bulma_workspace,
)
from opencas.legacy.models import (
    BulmaEpisode,
    BulmaMemoryEdge,
    BulmaHistoryEntry,
    BulmaIdentityProfile,
    BulmaSpark,
    BulmaWorkspaceManifest,
)
from opencas.memory.models import EpisodeKind
from opencas.somatic.models import PrimaryEmotion, SocialTarget


def test_map_bulma_episode_v3_source() -> None:
    be = BulmaEpisode.model_validate(
        {
            "id": "2026-02-19_0",
            "timestampMs": 1771485080000,
            "source": "v3:2026-02-19.md",
            "textContent": "Testing",
            "emotion": {
                "primaryEmotion": "neutral",
                "valence": 0,
                "arousal": 0.55,
                "certainty": 0.72,
                "emotionalIntensity": 0.55,
                "socialTarget": "other",
                "emotionTags": ["conversation", "testing"],
            },
            "salience": 0.62,
            "identityCore": False,
        }
    )
    ep = map_bulma_episode(be)
    assert ep.kind == EpisodeKind.OBSERVATION
    assert ep.session_id == "2026-02-19"
    assert ep.content == "Testing"
    assert ep.salience == 0.62
    assert ep.identity_core is False
    assert ep.affect is not None
    assert ep.affect.primary_emotion == PrimaryEmotion.NEUTRAL
    assert ep.affect.social_target == SocialTarget.OTHER
    assert ep.payload["bulma_id"] == "2026-02-19_0"


def test_map_bulma_episode_chat_source():
    be = BulmaEpisode.model_validate(
        {
            "id": "chat-1",
            "timestampMs": 1771485364000,
            "source": "chat",
            "textContent": "hello bulma",
            "emotion": {
                "primaryEmotion": "joy",
                "valence": 0.3,
                "arousal": 0.6,
                "certainty": 0.8,
                "emotionalIntensity": 0.6,
                "socialTarget": "user",
                "emotionTags": ["greeting"],
            },
            "salience": 0.75,
            "identityCore": True,
        }
    )
    ep = map_bulma_episode(be)
    assert ep.kind == EpisodeKind.TURN
    assert ep.identity_core is True
    assert ep.affect.primary_emotion == PrimaryEmotion.JOY
    assert ep.affect.social_target == SocialTarget.USER


def test_map_bulma_episode_uses_stable_id_not_timestamp() -> None:
    first = BulmaEpisode.model_validate(
        {
            "id": "ep-a",
            "timestampMs": 1771485364000,
            "source": "chat",
            "textContent": "first",
        }
    )
    second = BulmaEpisode.model_validate(
        {
            "id": "ep-b",
            "timestampMs": 1771485364000,
            "source": "chat",
            "textContent": "second",
        }
    )

    assert map_bulma_episode(first).episode_id != map_bulma_episode(second).episode_id
    assert map_bulma_episode(first).episode_id == bulma_episode_uuid("ep-a")


def test_map_bulma_episode_accepts_structured_encoded_metadata() -> None:
    episode = BulmaEpisode.model_validate(
        {
            "id": "ep-meta",
            "timestampMs": 1771485364000,
            "source": "chat",
            "textContent": "metadata",
            "metadata": {
                "v3": {
                    "source": "chat",
                    "encodedMetadata": {"topics": ["memory"], "importance": 0.9},
                    "extraField": {"kept": True},
                }
            },
        }
    )

    mapped = map_bulma_episode(episode)
    assert mapped.payload["bulma_v3"]["encodedMetadata"]["topics"] == ["memory"]
    assert mapped.payload["bulma_metadata"]["v3"]["extraField"] == {"kept": True}


def test_map_bulma_memory_edge_uses_episode_id_map() -> None:
    edge = BulmaMemoryEdge.model_validate(
        {
            "sourceId": "ep-a",
            "targetId": "ep-b",
            "semanticWeight": 0.9,
            "confidence": 0.8,
        }
    )

    mapped = map_bulma_memory_edge(
        edge,
        {"ep-a": str(bulma_episode_uuid("ep-a")), "ep-b": str(bulma_episode_uuid("ep-b"))},
    )

    assert mapped.source_id == str(bulma_episode_uuid("ep-a"))
    assert mapped.target_id == str(bulma_episode_uuid("ep-b"))


def test_map_bulma_identity() -> None:
    profile = BulmaIdentityProfile.model_validate(
        {
            "updatedAtMs": 1775417396065,
            "coreNarrative": "Bulma prioritizes continuity, care.",
            "values": ["continuity", "care"],
            "ongoingGoals": ["persistence", "memory", "assist"],
            "traits": ["curious", "patient"],
            "partner": {"userId": "jarrod", "trust": 0.92, "musubi": 1.0},
            "recentThemes": [],
            "memoryAnchors": [],
            "recentActivities": [],
        }
    )
    mapped = map_bulma_identity(profile)
    assert mapped["narrative"] == "Bulma prioritizes continuity, care."
    assert mapped["values"] == ["continuity", "care"]
    assert mapped["partner_user_id"] == "jarrod"
    assert mapped["partner_trust"] == 0.92
    assert mapped["partner_musubi"] == 1.0


def test_map_bulma_spark() -> None:
    bs = BulmaSpark.model_validate(
        {
            "id": "spark-1",
            "timestampMs": 1773554000251,
            "mode": "reverie",
            "trigger": "manual",
            "interest": "odd online communities",
            "summary": "settling into the quiet",
            "label": "Research: next leads",
            "kind": "research",
            "intensity": 0.41,
            "objective": "Research this curiosity thread",
            "tags": ["daydream-spark", "research"],
        }
    )
    dr = map_bulma_spark(bs)
    assert "odd online communities" in dr.spark_content
    assert dr.synthesis == "Research this curiosity thread"
    assert dr.keeper is False


def test_map_bulma_history() -> None:
    entry = BulmaHistoryEntry.model_validate(
        {
            "timestampMs": 1773554000249,
            "mode": "reverie",
            "trigger": "manual",
            "boredom": 0.35,
            "motivation": 0.46,
            "interest": "odd online communities",
            "summary": "settling into the quiet",
            "tags": ["reverie", "inner-life"],
        }
    )
    dr = map_bulma_history(entry)
    assert dr.created_at == datetime.fromtimestamp(1773554000249 / 1000.0, tz=timezone.utc)


def test_map_bulma_workspace_project() -> None:
    manifest = BulmaWorkspaceManifest(
        project_name="openbulma-v3",
        source_dir="/tmp/bulma/workspaces/openbulma-v3",
        files=["a.md", "b.py", "c.txt", "d.json"],
        meta={"has_subdirectories": True},
    )
    work = map_bulma_workspace(manifest)
    from opencas.autonomy.models import WorkStage

    assert work.stage == WorkStage.PROJECT
    assert "openbulma-v3" in work.content


def test_map_bulma_workspace_artifact() -> None:
    manifest = BulmaWorkspaceManifest(
        project_name="single-artifact",
        source_dir="/tmp/bulma/workspaces/single-artifact",
        files=["readme.md"],
        meta={"has_subdirectories": False},
    )
    work = map_bulma_workspace(manifest)
    from opencas.autonomy.models import WorkStage

    assert work.stage == WorkStage.ARTIFACT
