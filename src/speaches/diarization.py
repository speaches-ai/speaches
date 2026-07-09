from collections.abc import Hashable

import numpy as np
from pydantic import BaseModel, ConfigDict

from speaches.audio import Audio


class KnownSpeaker(BaseModel):
    name: str
    audio: Audio

    model_config = ConfigDict(arbitrary_types_allowed=True)


def match_speakers_to_known(
    avg_embeddings: dict[Hashable, np.ndarray],
    known_embeddings: dict[str, np.ndarray],
    threshold: float,
) -> dict[Hashable, str]:
    # Map each diarized speaker to the most similar known speaker via cosine similarity.
    # A diarized speaker is only relabeled when its best match strictly exceeds the threshold;
    # otherwise it keeps its original label (e.g. SPEAKER_00). This makes known speakers
    # optional hints: their count does not force additional speakers and unmatched speakers
    # are not force-labeled with the nearest name.
    mapping: dict[Hashable, str] = {}
    for diarized_spk, diarized_emb in avg_embeddings.items():
        best_name = diarized_spk
        best_sim = threshold
        for known_name, known_emb in known_embeddings.items():
            denom = float(np.linalg.norm(diarized_emb) * np.linalg.norm(known_emb))
            if denom < 1e-8:
                continue
            sim = float(np.dot(diarized_emb, known_emb) / denom)
            if sim > best_sim:
                best_sim = sim
                best_name = known_name
        mapping[diarized_spk] = best_name

    return mapping
