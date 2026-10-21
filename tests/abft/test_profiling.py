"""Check profiling attribution without presenting CPU fixtures as GPU timings."""
from types import SimpleNamespace

import pytest

from abft.profiling import STAGES, summarize_profile


def event(name, *, device="CPU", gpu=10.0, cpu=30.0):
    return SimpleNamespace(name=name, device_type=SimpleNamespace(name=device),
                           device_time_total=gpu, cpu_time_total=cpu)


def samples():
    return [event(name) for _ in range(2) for name in STAGES]


def test_profile_excludes_warmup_cuda_annotations_and_duplicate_parent():
    values = samples() + [event("abft.execute", device="CUDA", gpu=9000),
                          event("warmup.abft.execute", gpu=9000)]
    result = summarize_profile(values, 2, input_fused=True)
    rows = {row["stage"]: row for row in result["stages"]}
    assert "abft.verify" not in rows
    assert "abft.detect" in rows and "abft.locate" in rows
    assert rows["abft.execute"]["gpu_active_ms"]["samples"] == [.01, .01]
    assert rows["abft.execute"]["cpu_scope_ms"]["samples"] == [.03, .03]
    assert "compilation" in result["exclusions"]
    assert "printing" in result["exclusions"]


def test_generic_plugin_and_separate_reference_remain_profileable():
    values = [e for e in samples() if e.name not in {"abft.detect", "abft.locate", "abft.merge_reference"}]
    result = summarize_profile(values, 2, input_fused=False)
    assert "abft.verify" in {row["stage"] for row in result["stages"]}


def test_missing_samples_or_gpu_activity_fail_explicitly():
    with pytest.raises(RuntimeError, match="Incomplete profile"):
        summarize_profile(samples()[1:], 2, input_fused=True)
    values = samples()
    for e in values:
        e.device_time_total = 0
    with pytest.raises(RuntimeError, match="CUDA activity timings are unavailable"):
        summarize_profile(values, 2, input_fused=True)


@pytest.mark.parametrize("invalid", [-1, float("nan"), float("inf")])
def test_invalid_stage_time_is_not_reported_as_success(invalid):
    values = samples()
    values[1].device_time_total = invalid
    with pytest.raises(ValueError, match="finite and nonnegative"):
        summarize_profile(values, 2, input_fused=True)
