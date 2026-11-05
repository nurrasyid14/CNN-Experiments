from __future__ import annotations

import copy
from collections import Counter
from time import perf_counter
from typing import Callable, Iterable

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support


class MetricsCounter:
    """Accumulate scalar metric observations."""

    def __init__(self) -> None:
        self._counts: Counter[str] = Counter()
        self._totals: dict[str, float] = {}

    def update(self, metrics: dict[str, float | int]) -> None:
        for name, value in metrics.items():
            numeric_value = float(value)
            self._counts[name] += 1
            self._totals[name] = self._totals.get(name, 0.0) + numeric_value

    def update_many(self, name: str, values: Iterable[float | int]) -> None:
        for value in values:
            self.update({name: value})

    def mean(self, name: str) -> float | None:
        count = self._counts[name]
        return self._totals[name] / count if count else None

    def totals(self) -> dict[str, float]:
        return dict(self._totals)

    def counts(self) -> dict[str, int]:
        return dict(self._counts)


def classification_metrics(
    targets: Iterable[int],
    predictions: Iterable[int],
    labels: list[str] | None = None,
) -> dict[str, object]:
    target_array = np.asarray(list(targets))
    prediction_array = np.asarray(list(predictions))
    precision, recall, f1, support = precision_recall_fscore_support(
        target_array,
        prediction_array,
        average=None,
        labels=np.unique(target_array),
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        target_array,
        prediction_array,
        average="macro",
        zero_division=0,
    )
    class_ids = np.unique(target_array)
    names = labels if labels is not None else [str(value) for value in class_ids]
    if len(names) != len(class_ids):
        names = [str(value) for value in class_ids]
    return {
        "accuracy": float(np.mean(target_array == prediction_array)),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(names)
        },
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
    }


def count_model_parameters(model: torch.nn.Module) -> dict[str, int]:
    trainable = sum(parameter.numel() for parameter in model.parameters()
                    if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return {"total": total, "trainable": trainable, "non_trainable": total - trainable}


def estimate_model_flops(
    model: torch.nn.Module,
    sample: torch.Tensor,
) -> int | None:
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if sample.is_cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, with_flops=True) as profile:
            with torch.inference_mode():
                model(sample)
        return sum(getattr(event, "flops", 0) for event in profile.key_averages())
    except (AttributeError, RuntimeError, ValueError) as error:
        print(f"FLOPs unavailable: {error}")
        return None


def peak_gpu_memory_mib(device: torch.device) -> float | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    torch.cuda.synchronize(device)
    return float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))


def reset_peak_gpu_memory(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def benchmark_throughput(
    operation: Callable[[], None],
    iterations: int,
    device: torch.device,
) -> dict[str, float]:
    for _ in range(min(10, iterations)):
        operation()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = perf_counter()
    for _ in range(iterations):
        operation()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = perf_counter() - start
    return {
        "iterations": float(iterations),
        "elapsed_seconds": elapsed_seconds,
        "operations_per_second": iterations / elapsed_seconds,
    }


def compare_preprocessing_execution(
    cpu_operation: Callable[[], None],
    cuda_operation: Callable[[], None] | None,
    iterations: int,
) -> dict[str, object]:
    results: dict[str, object] = {
        "cpu": benchmark_throughput(
            cpu_operation,
            iterations,
            torch.device("cpu"),
        ),
    }
    if cuda_operation is None or not torch.cuda.is_available():
        results["cuda"] = {"status": "unavailable"}
        return results
    results["cuda"] = benchmark_throughput(
        cuda_operation,
        iterations,
        torch.device("cuda"),
    )
    return results


def benchmark_inference(
    model: torch.nn.Module,
    sample: torch.Tensor,
    iterations: int = 100,
    warmup: int = 10,
) -> dict[str, float]:
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        if sample.is_cuda:
            torch.cuda.synchronize(sample.device)
        latencies_ms = []
        for _ in range(iterations):
            start = perf_counter()
            model(sample)
            if sample.is_cuda:
                torch.cuda.synchronize(sample.device)
            latencies_ms.append((perf_counter() - start) * 1000.0)
    values = np.asarray(latencies_ms)
    return {
        "warmup": float(warmup),
        "iterations": float(iterations),
        "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "samples_per_second": float(1000.0 / values.mean() * sample.shape[0]),
    }


def compare_model_execution(
    model: torch.nn.Module,
    sample_shape: tuple[int, ...],
    iterations: int = 100,
) -> dict[str, object]:
    cpu_model = copy.deepcopy(model).cpu().eval()
    cpu_sample = torch.zeros(sample_shape, dtype=torch.float32, device="cpu")
    results: dict[str, object] = {
        "cpu": benchmark_inference(cpu_model, cpu_sample, iterations=iterations)
    }
    if not torch.cuda.is_available():
        results["cuda"] = {"status": "unavailable"}
        return results
    cuda_model = copy.deepcopy(model).cuda().eval()
    cuda_sample = torch.zeros(sample_shape, dtype=torch.float32, device="cuda")
    results["cuda"] = benchmark_inference(cuda_model, cuda_sample, iterations=iterations)
    return results
