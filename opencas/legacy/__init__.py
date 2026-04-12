"""Legacy import adapters for OpenBulma v4 state migration into OpenCAS."""

from .importer import BulmaImportTask, ImportReport
from .cutover import preflight_bulma_state
from .loader import stream_jsonl, load_json
from .mapper import map_bulma_episode, map_bulma_identity
from .models import BulmaEpisode, BulmaIdentityProfile, BulmaSpark

__all__ = [
    "BulmaImportTask",
    "ImportReport",
    "preflight_bulma_state",
    "stream_jsonl",
    "load_json",
    "map_bulma_episode",
    "map_bulma_identity",
    "BulmaEpisode",
    "BulmaIdentityProfile",
    "BulmaSpark",
]
