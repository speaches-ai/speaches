from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class SileroVADModelV6:
    def __init__(self, model_path: Path, providers: list[tuple[str, dict]], sess_options: object | None = None) -> None:
        import onnxruntime

        if sess_options is None:
            sess_options = onnxruntime.SessionOptions()

        self.session = onnxruntime.InferenceSession(
            model_path,
            providers=providers,
            sess_options=sess_options,
        )

    def __call__(
        self, audio: np.ndarray, num_samples: int = 512, context_size_samples: int = 64
    ) -> NDArray[np.float32]:
        timelog_start = time.perf_counter()

        if audio.ndim == 2:
            assert audio.shape[0] == 1, "Batched inference (batch > 1) is not supported for v6"
            audio = audio.squeeze(0)
        assert audio.ndim == 1, "Input should be a 1D array"
        assert audio.shape[0] % num_samples == 0, "Input size should be a multiple of num_samples"

        h = np.zeros((1, 1, 128), dtype=np.float32)
        c = np.zeros((1, 1, 128), dtype=np.float32)

        batched_audio = audio.reshape(-1, num_samples)
        context = batched_audio[..., -context_size_samples:]
        context[-1] = 0
        context = np.roll(context, 1, 0)
        batched_audio = np.concatenate([context, batched_audio], 1)

        encoder_batch_size = 10000
        outputs = []
        for i in range(0, batched_audio.shape[0], encoder_batch_size):
            output, h, c = self.session.run(
                None,
                {"input": batched_audio[i : i + encoder_batch_size], "h": h, "c": c},
            )
            outputs.append(output)

        out = np.concatenate(outputs, axis=0)
        logger.debug(f"VAD v6 model inference took {time.perf_counter() - timelog_start:.4f}s")
        # Return 2D (1, num_windows) to match v5 interface
        return out.reshape(1, -1)
