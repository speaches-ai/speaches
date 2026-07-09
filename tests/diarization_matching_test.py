import numpy as np

from speaches.diarization import match_speakers_to_known


def test_match_names_matching_reference() -> None:
    known = {
        "alice": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "bob": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    }
    avg = {"SPEAKER_00": np.array([0.99, 0.01, 0.0], dtype=np.float32)}

    mapping = match_speakers_to_known(avg, known, threshold=0.5)

    assert mapping == {"SPEAKER_00": "alice"}


def test_match_keeps_label_when_below_threshold() -> None:
    known = {
        "alice": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "bob": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    }
    # Orthogonal to every reference, so cosine similarity is 0, below the threshold.
    avg = {"SPEAKER_00": np.array([0.0, 0.0, 1.0], dtype=np.float32)}

    mapping = match_speakers_to_known(avg, known, threshold=0.5)

    assert mapping == {"SPEAKER_00": "SPEAKER_00"}


def test_extra_references_do_not_add_or_rename_speakers() -> None:
    # A catalog of references, only one of which is actually present in the audio.
    known = {
        "alice": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "bob": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        "carol": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        "dave": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    }
    # Two real speakers: one is alice, the other resembles none of the references.
    avg = {
        "SPEAKER_00": np.array([1.0, 0.02, 0.0, 0.0], dtype=np.float32),
        "SPEAKER_01": np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
    }

    mapping = match_speakers_to_known(avg, known, threshold=0.8)

    # The speaker count is unchanged (one entry per diarized speaker) and the unmatched
    # speaker keeps its label rather than being force-named from the catalog.
    assert len(mapping) == 2
    assert mapping["SPEAKER_00"] == "alice"
    assert mapping["SPEAKER_01"] == "SPEAKER_01"


def test_threshold_gates_the_match() -> None:
    known = {"alice": np.array([1.0, 0.0], dtype=np.float32)}
    # Unit vector whose cosine similarity with "alice" is about 0.6.
    avg = {"SPEAKER_00": np.array([0.6, 0.8], dtype=np.float32)}

    # A threshold above the similarity keeps the label; one below names the speaker.
    assert match_speakers_to_known(avg, known, threshold=0.7) == {"SPEAKER_00": "SPEAKER_00"}
    assert match_speakers_to_known(avg, known, threshold=0.5) == {"SPEAKER_00": "alice"}


def test_zero_embedding_is_skipped() -> None:
    known = {"alice": np.array([0.0, 0.0], dtype=np.float32)}
    avg = {"SPEAKER_00": np.array([1.0, 0.0], dtype=np.float32)}

    # A degenerate zero-norm reference cannot match; the speaker keeps its label.
    assert match_speakers_to_known(avg, known, threshold=0.5) == {"SPEAKER_00": "SPEAKER_00"}
