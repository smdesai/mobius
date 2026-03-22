"""Measure prediction latency by running the model with random inputs."""

from __future__ import annotations

import time
from functools import reduce
from pathlib import Path
from typing import Any

import numpy as np

import CoreML
from Foundation import NSURL

from .compute_plan import COMPUTE_UNITS


def _fill_multiarray(ml_array: Any, shape: tuple[int, ...], dtype: Any) -> None:
    """Fill MLMultiArray with random data by iterating over indices."""
    total = reduce(lambda a, b: a * b, shape, 1)
    is_int = dtype in (np.int32, np.int64)
    for i in range(total):
        # Compute multi-dimensional index
        idx = []
        remaining = i
        for s in reversed(shape):
            idx.insert(0, remaining % s)
            remaining //= s
        val = int(np.random.randint(0, 100)) if is_int else float(np.random.randn())
        ml_array.setObject_atIndexedSubscript_(val, i)


def _infer_length_value(name: str, tensor_shapes: dict[str, tuple[int, ...]]) -> int | None:
    """For a length-like input, find the matching tensor and return its sequence dim."""
    base = name
    for suffix in ("_length", "_len", "length", "len"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    else:
        return None

    base = base.rstrip("_")
    if not base:
        return None

    for candidate in (base, base + "s", base.rstrip("s")):
        if candidate in tensor_shapes:
            shape = tensor_shapes[candidate]
            if len(shape) >= 2:
                return shape[1]
            elif len(shape) == 1:
                return shape[0]
    return None


def _make_input_provider(model_desc: Any) -> tuple[Any, bool]:
    """Create an MLDictionaryFeatureProvider with random data for all inputs.

    Returns (provider, has_state) where has_state indicates the model uses MLState.
    """
    input_desc = model_desc.inputDescriptionsByName()
    input_dict = {}
    has_state = False
    tensor_shapes: dict[str, tuple[int, ...]] = {}

    for name in input_desc:
        feat = input_desc[name]
        feat_type = feat.type()

        if feat_type == CoreML.MLFeatureTypeState:
            has_state = True
            continue  # State is passed via MLState, not the input dict

        if feat_type == CoreML.MLFeatureTypeMultiArray:
            constraint = feat.multiArrayConstraint()
            shape = tuple(int(d) for d in constraint.shape())
            ml_dtype = constraint.dataType()

            dtype_map = {
                CoreML.MLMultiArrayDataTypeFloat16: np.float16,
                CoreML.MLMultiArrayDataTypeFloat32: np.float32,
                CoreML.MLMultiArrayDataTypeFloat64: np.float64,
                CoreML.MLMultiArrayDataTypeInt32: np.int32,
            }
            np_dtype = dtype_map.get(ml_dtype, np.float32)

            ml_array, err = CoreML.MLMultiArray.alloc().initWithShape_dataType_error_(
                list(shape), ml_dtype, None
            )
            if err:
                raise RuntimeError(f"Failed to create MLMultiArray for '{name}': {err}")

            _fill_multiarray(ml_array, shape, np_dtype)
            input_dict[name] = CoreML.MLFeatureValue.featureValueWithMultiArray_(ml_array)
            tensor_shapes[name] = shape

    # Fix length-like scalar inputs to match their corresponding tensor dimension
    for name, fv in input_dict.items():
        shape = tensor_shapes.get(name)
        if shape is None:
            continue
        total = reduce(lambda a, b: a * b, shape, 1)
        if total != 1:
            continue
        length_val = _infer_length_value(name, tensor_shapes)
        if length_val is not None:
            ml_array = fv.multiArrayValue()
            ml_array.setObject_atIndexedSubscript_(length_val, 0)

    provider, err = CoreML.MLDictionaryFeatureProvider.alloc().initWithDictionary_error_(
        input_dict, None
    )
    if err:
        raise RuntimeError(f"Failed to create input provider: {err}")
    return provider, has_state


def _compute_stats(times_ms: list[float]) -> dict:
    if not times_ms:
        return {"median_ms": 0.0, "mean_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "std_ms": 0.0}
    times_ms.sort()
    n = len(times_ms)
    mean = sum(times_ms) / n
    median = times_ms[n // 2] if n % 2 else (times_ms[n // 2 - 1] + times_ms[n // 2]) / 2
    variance = sum((t - mean) ** 2 for t in times_ms) / n
    return {
        "median_ms": round(median, 3),
        "mean_ms": round(mean, 3),
        "min_ms": round(times_ms[0], 3),
        "max_ms": round(times_ms[-1], 3),
        "std_ms": round(variance ** 0.5, 3),
    }


def measure_latency(
    model_path: Path,
    compute_units: str,
    warmup: int = 5,
    iterations: int = 10,
) -> dict:
    """Load model via PyObjC and measure compile + prediction latency.

    Returns dict with compile_ms and prediction stats.
    """
    url = NSURL.fileURLWithPath_(str(model_path))
    config = CoreML.MLModelConfiguration.alloc().init()
    config.setComputeUnits_(COMPUTE_UNITS[compute_units])

    # Measure compile/load time
    compile_start = time.perf_counter()
    model, error = CoreML.MLModel.modelWithContentsOfURL_configuration_error_(url, config, None)
    compile_ms = (time.perf_counter() - compile_start) * 1000

    if error or model is None:
        return {"error": str(error) if error else "failed to load model"}

    model_desc = model.modelDescription()

    try:
        provider, has_state = _make_input_provider(model_desc)
    except Exception as e:
        return {"error": f"failed to create inputs: {e}"}

    state = None
    if has_state:
        state = model.makeState()

    def _predict():
        if state is not None:
            return model.predictionFromFeatures_usingState_error_(provider, state, None)
        return model.predictionFromFeatures_error_(provider, None)

    # Warmup
    for _ in range(warmup):
        result, err = _predict()
        if err:
            return {
                "compile_ms": round(compile_ms, 3),
                "error": f"prediction failed: {err}",
            }

    # Timed runs
    times_ms = []
    for _ in range(iterations):
        start = time.perf_counter()
        _predict()
        elapsed = (time.perf_counter() - start) * 1000
        times_ms.append(elapsed)

    stats = _compute_stats(times_ms)
    return {
        "compile_ms": round(compile_ms, 3),
        **stats,
        "iterations": iterations,
    }
