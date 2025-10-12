from collections.abc import Generator
import logging
from typing import TypedDict

import numpy as np
import onnxruntime as ort
from pyannote.audio import Model
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from rich.console import Console
from rich.table import Table
import torch

from pyannote_embedding import PYANNOTE_EMBEDDING_ONNX_PATH, PYANNOTE_EMBEDDING_TORCH_MODEL_NAME, SAMPLE_RATE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TODO: test performance separatly on different ONNX providers (CPU, CUDA, etc.)


class AccuracyStats(TypedDict):
    duration: float
    max_diff: float
    mean_diff: float
    std_diff: float
    cosine_sim: float
    rel_error: float
    exceeds_tolerance: bool


# Constants
TOLERANCE = 1e-3
DURATIONS = [1, 2, 4, 8, 16, 32, 64]  # 1s to 64s


@pytest.fixture(scope="session")
def onnx_model_path() -> str:
    if not PYANNOTE_EMBEDDING_ONNX_PATH.exists():
        pytest.skip(f"ONNX model not found at: {PYANNOTE_EMBEDDING_ONNX_PATH}. Run the export script first.")
    return str(PYANNOTE_EMBEDDING_ONNX_PATH)


@pytest.fixture(scope="session")
def pytorch_model() -> Model:
    model = Model.from_pretrained(PYANNOTE_EMBEDDING_TORCH_MODEL_NAME)
    assert model is not None, "Failed to load PyTorch model"
    model.eval()
    return model


@pytest.fixture(scope="session")
def ort_session(onnx_model_path: str) -> ort.InferenceSession:
    return ort.InferenceSession(onnx_model_path)


@pytest.mark.parametrize("duration", DURATIONS)
def test_onnx_model_inference(ort_session: ort.InferenceSession, duration: float) -> None:
    logger.info(f"Testing ONNX inference with {duration}s audio")

    input_info = ort_session.get_inputs()[0]
    num_samples = int(SAMPLE_RATE * duration)
    rng = np.random.default_rng()
    test_waveform = rng.standard_normal((1, num_samples)).astype(np.float32)

    ort_inputs = {input_info.name: test_waveform}
    ort_outputs = ort_session.run(None, ort_inputs)
    embeddings = ort_outputs[0]

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 1, f"Expected batch size 1, got {embeddings.shape[0]}"
    assert embeddings.shape[1] == 512, f"Expected 512 dimensions, got {embeddings.shape[1]}"
    assert not np.isnan(embeddings).any(), "Output contains NaN values"
    assert not np.isinf(embeddings).any(), "Output contains infinite values"


@pytest.fixture(scope="session")
def accuracy_collector() -> list[AccuracyStats]:
    return []


@pytest.fixture(scope="session", autouse=True)
def accuracy_reporter(accuracy_collector: list[AccuracyStats]) -> Generator[None]:
    yield  # All tests run here
    # Display table after all tests complete
    if accuracy_collector:
        console = Console()

        # Create rich table
        table = Table(title="Model Accuracy Comparison Statistics")
        table.add_column("Duration", style="cyan", width=10)
        table.add_column("Max Diff", style="magenta", width=12)
        table.add_column("Mean Diff", style="blue", width=12)
        table.add_column("Std Diff", style="blue", width=12)
        table.add_column("Cosine Sim", style="green", width=12)
        table.add_column("Rel Error", style="yellow", width=12)
        table.add_column("Exceeds Tol", style="red", width=12)

        # Sort by duration and add rows
        exceeding_count = 0
        for stat in sorted(accuracy_collector, key=lambda x: x["duration"]):
            exceeds = stat["exceeds_tolerance"]
            if exceeds:
                exceeding_count += 1

            table.add_row(
                f"{stat['duration']:.0f}s",
                f"{stat['max_diff']:.8f}",
                f"{stat['mean_diff']:.8f}",
                f"{stat['std_diff']:.8f}",
                f"{stat['cosine_sim']:.6f}",
                f"{stat['rel_error']:.8f}",
                "[red]YES[/red]" if exceeds else "[green]NO[/green]",
            )

        # Display table
        console.print()
        console.print(table)
        console.print()
        console.print(f"[bold]Tolerance threshold:[/bold] {TOLERANCE}")
        console.print(f"[bold]Durations exceeding tolerance:[/bold] {exceeding_count}/{len(accuracy_collector)}")
        console.print()


@pytest.mark.parametrize("duration", DURATIONS)
def test_model_accuracy_comparison(
    pytorch_model: Model, ort_session: ort.InferenceSession, duration: float, accuracy_collector: list[AccuracyStats]
) -> None:
    logger.info(f"Testing accuracy comparison for {duration}s audio")

    num_samples = int(SAMPLE_RATE * duration)
    test_waveform_torch = torch.randn(1, num_samples)
    test_waveform_numpy = test_waveform_torch.numpy().astype(np.float32)
    ort_inputs = {ort_session.get_inputs()[0].name: test_waveform_numpy}

    # Get outputs from both models
    with torch.no_grad():
        pytorch_output = pytorch_model(test_waveform_torch)

    onnx_output = ort_session.run(None, ort_inputs)[0]

    # Calculate statistics
    diff = np.abs(pytorch_output.numpy() - onnx_output)
    max_diff = diff.max()
    mean_diff = diff.mean()
    std_diff = diff.std()

    # Calculate cosine similarity
    pytorch_norm = pytorch_output.numpy() / np.linalg.norm(pytorch_output.numpy())
    onnx_norm = onnx_output / np.linalg.norm(onnx_output)  # pyright: ignore[reportArgumentType, reportCallIssue]
    cosine_sim = np.dot(pytorch_norm.flatten(), onnx_norm.flatten())

    # Calculate relative error
    rel_error = max_diff / np.abs(pytorch_output.numpy()).max()

    # Collect data for summary table
    accuracy_collector.append(
        {
            "duration": duration,
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "std_diff": std_diff,
            "cosine_sim": cosine_sim,
            "rel_error": rel_error,
            "exceeds_tolerance": max_diff > TOLERANCE,
        }
    )

    logger.info(f"Max absolute difference: {max_diff:.8f}")
    logger.info(f"Cosine similarity: {cosine_sim:.6f}")


@pytest.mark.parametrize("duration", DURATIONS)
def test_pytorch_model_benchmark(benchmark: BenchmarkFixture, pytorch_model: Model, duration: float) -> None:
    num_samples = int(SAMPLE_RATE * duration)
    test_waveform = torch.randn(1, num_samples)

    def pytorch_inference() -> torch.Tensor:
        with torch.no_grad():
            return pytorch_model(test_waveform)

    benchmark.name = f"pytorch_{duration}s"
    benchmark.group = "pytorch_inference"
    result = benchmark(pytorch_inference)
    assert result.shape == (1, 512)


@pytest.mark.parametrize("duration", DURATIONS)
def test_onnx_model_benchmark(benchmark: BenchmarkFixture, ort_session: ort.InferenceSession, duration: float) -> None:
    num_samples = int(SAMPLE_RATE * duration)
    input_name = ort_session.get_inputs()[0].name
    rng = np.random.default_rng()
    test_waveform = rng.standard_normal((1, num_samples)).astype(np.float32)
    ort_inputs = {input_name: test_waveform}

    def onnx_inference() -> np.ndarray:
        return ort_session.run(None, ort_inputs)[0]  # pyright: ignore[reportReturnType]

    benchmark.name = f"onnx_{duration}s"
    benchmark.group = "onnx_inference"
    result = benchmark(onnx_inference)
    assert result.shape == (1, 512)
