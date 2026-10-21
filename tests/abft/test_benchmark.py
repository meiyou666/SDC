"""CPU tests cover scheduling/configuration, never claim GPU performance."""
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from abft.benchmark import load_config, measure_pairs, summarize


@pytest.fixture
def benchmark_yaml():
    path = Path(__file__).resolve().parents[2] / "configs/gemm/abft_benchmark.yaml"
    return yaml.safe_load(path.read_text())


def write_config(tmp_path, values):
    path = tmp_path / "benchmark.yaml"
    path.write_text(yaml.safe_dump(values))
    return path


def test_all_native_formats_and_explicit_parameters(tmp_path, benchmark_yaml):
    config = load_config(write_config(tmp_path, benchmark_yaml))
    assert {fmt["precision"] for fmt in config.formats} == {"fp32", "bf16", "fp8", "nvfp4"}
    assert config.abft.on_error == "report" and config.abft.recovery == "none"
    assert config.abft.timing is False
    assert config.profile_repeats == 0
    assert config.progress is False and config.preview_size == 0


@pytest.mark.parametrize(("field", "value"), [
    ("device", "cpu"), ("device", "cuda"), ("repeats", 1), ("warmup", 0),
    ("profile_repeats", -1), ("input_std", float("nan")), ("input_std", True),
    ("seeds", [-1]), ("shapes", [[128, 128, 15]]), ("output", ""),
    ("shapes", [[128, 128, 16]]),
    ("progress", "true"), ("preview_size", -1), ("preview_size", 17),
    ("preview_size", True), ("preview_size", 1.5),
])
def test_reject_invalid_experiments(tmp_path, benchmark_yaml, field, value):
    benchmark_yaml[field] = value
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path, benchmark_yaml))


@pytest.mark.parametrize(("field", "value"), [
    ("method", "none"), ("on_error", "raise"), ("recovery", "recompute"), ("timing", True),
])
def test_reject_sync_or_profile_in_main_measurement(tmp_path, benchmark_yaml, field, value):
    benchmark_yaml["abft"][field] = value
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path, benchmark_yaml))


def test_format_validation_and_unknown_config(tmp_path, benchmark_yaml):
    for change in ({"global_scale": -1}, {"scaling": "block16"}, {"precision": "fp16"}):
        values = deepcopy(benchmark_yaml)
        values["formats"][0].update(change)
        with pytest.raises(ValueError):
            load_config(write_config(tmp_path, values))
    benchmark_yaml["fault"] = {"location": "a"}
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path, benchmark_yaml))


@pytest.mark.parametrize("progress", [False, True])
def test_paired_schedule_and_observation_boundary(progress):
    history = []
    active = False
    shared_prepared = object()

    def operation(name):
        def call():
            assert active
            history.append(name)
            return shared_prepared
        return call

    def measure(fn):
        nonlocal active
        active = True
        value = fn()
        active = False
        return value, 2.0 if history[-1] == "baseline" else 6.0

    def observe(baseline, protected):
        assert not active
        assert baseline is protected is shared_prepared
        history.append("observe")
        return {"detected": False}

    notifications = []

    def on_pair(index, baseline_ms, protected_ms, observation):
        assert not active
        assert history[-1] == "observe"
        assert observation == {"detected": False}
        notifications.append((index, baseline_ms, protected_ms))

    result = measure_pairs(operation("baseline"), operation("protected"), warmup=2, repeats=3,
                           measure=measure, observe=observe, on_pair=on_pair if progress else None)
    assert history == ["baseline", "protected", "protected", "baseline",
                       "baseline", "protected", "observe", "protected", "baseline", "observe",
                       "baseline", "protected", "observe"]
    assert result["paired_ratio"]["samples"] == [3.0, 3.0, 3.0]
    assert len(result["observations"]) == 3
    assert notifications == ([(1, 2.0, 6.0), (2, 2.0, 6.0), (3, 2.0, 6.0)] if progress else [])


def test_matrix_preview_is_bounded_and_optional(capsys):
    import torch

    from abft.benchmark import show_matrix

    matrix = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    show_matrix("A", matrix, 0)
    assert capsys.readouterr().out == ""
    show_matrix("A", matrix, 2)
    lines = capsys.readouterr().out.splitlines()
    assert "shape=(4, 5)" in lines[0]
    assert [[float(value) for value in line.split()] for line in lines[1:]] == [[0, 1], [5, 6]]


def test_statistics_reject_invalid_measurements():
    assert summarize([1.0, 2.0, 9.0])["median"] == 2.0
    for values in ([], [0.0], [-1.0], [float("inf")], [float("nan")]):
        with pytest.raises(ValueError):
            summarize(values)


@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
def test_bitwise_output_comparison(dtype_name):
    import torch

    from abft.benchmark import compare_outputs

    dtype = getattr(torch, dtype_name)
    positive = torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=dtype).T
    negative = positive.clone()
    negative[0, 0] = -0.0
    record = compare_outputs(positive, negative)
    assert record["output_equal"] and not record["output_bits_equal"]
    assert record["max_abs_finite_difference"] == 0
    nan = torch.full((1, 1), float("nan"), dtype=dtype)
    record = compare_outputs(nan, nan.clone())
    assert record["output_bits_equal"] and not record["output_equal"]
    assert record["nonfinite_elements"] == 1
    assert record["max_abs_finite_difference"] is None
