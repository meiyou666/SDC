from __future__ import annotations

import gc
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from .adapters import ModelAdapter
from .artifacts import (
    atomic_write_artifact,
    load_artifact,
    sha256_file,
    utc_now,
)
from .campaign import (
    MAIN_MODES,
    PAIR_ORDER,
    PilotCampaign,
    _file_records,
    _relative,
    pair_id,
)
from .manifest import (
    build_pair_manifest,
    load_manifest,
    manifest_sha256,
    save_manifest,
)
from .reduced_campaign import (
    FORMAL_OUTPUT_ROOT,
    PILOT_OUTPUT_ROOT,
    PROMPTS_PER_PAIR,
    ReducedFormalCampaign,
)
from .schema import Bounds, BoundsSource, FaultSpec
from .tasks import (
    DATASETS,
    MODELS,
    MODEL_DATASETS,
    REPO_ROOT,
    author_source_metadata,
)


DEFAULT_MAIN18K_ROOT = (
    REPO_ROOT / "reproduction" / "results" / "main_18k_v1"
)
VALIDATION_PATH = (
    REPO_ROOT
    / "reproduction"
    / "results"
    / "equivalence_validation_v1"
    / "validation.json"
)
TRIALS_PER_PROMPT_PER_FAULT_TYPE = 40
SPECS_PER_PAIR = 1200
TOTAL_FAULT_SPECS = 6000
TOTAL_FAULT_RUNS = 18000
TOTAL_CONTROLS = 150
ACCEPTED_OUTCOMES = {
    "MASKED_IDENTICAL",
    "MASKED_SEMANTIC",
    "SDC",
    "NON_EVALUABLE_CLEAN_FAILURE",
}


class Main18kCampaign(ReducedFormalCampaign):
    """Fresh, single-fingerprint 18,000-run paper-semantics campaign."""

    def __init__(
        self,
        output_root: str | Path = DEFAULT_MAIN18K_ROOT,
        *,
        formal_root: str | Path = FORMAL_OUTPUT_ROOT,
        pilot_root: str | Path = PILOT_OUTPUT_ROOT,
        validation_path: str | Path = VALIDATION_PATH,
    ):
        self.formal_root = Path(formal_root).resolve()
        self.validation_path = Path(validation_path).resolve()
        self.parent_campaign = load_artifact(
            self.formal_root / "campaign.json"
        )
        self.parent_status = load_artifact(
            self.formal_root / "main_status.json"
        )
        self.parent_selection = load_artifact(
            self.formal_root / "selection.json"
        )
        self.validation = load_artifact(self.validation_path)
        parent_fp = self.parent_campaign["campaign_fingerprint"]
        if self.parent_status.get("campaign_fingerprint") != parent_fp:
            raise RuntimeError("Parent main status fingerprint mismatch")
        if self.parent_selection.get("campaign_fingerprint") != parent_fp:
            raise RuntimeError("Parent selection fingerprint mismatch")
        if self.validation.get("status") != "passed":
            raise RuntimeError("Equivalence validation has not passed")
        super().__init__(
            output_root=output_root,
            pilot_root=pilot_root,
        )

    def _validate_frozen_config(self) -> None:
        PilotCampaign._validate_frozen_config(self)
        if len(MAIN_MODES) != 3:
            raise ValueError("The main campaign must have exactly three modes")
        mode_ids = [item.mode_id for item in MAIN_MODES]
        if mode_ids != [
            "no_protection",
            "paper_clamp_first_token_bounds",
            "paper_clamp_offline_bounds",
        ]:
            raise ValueError("Main protection modes changed")

    def _immutable_identity(self) -> dict[str, Any]:
        identity = PilotCampaign._immutable_identity(self)
        identity["schema_version"] = 3
        identity["experiment_id"] = "ft2_rtx4090_main_18k_v1"
        identity.pop("pilot_dimensions", None)
        identity["main_dimensions"] = {
            "pairs": 5,
            "prompts_per_pair": PROMPTS_PER_PAIR,
            "fault_types": 3,
            "trials_per_prompt_per_fault_type":
                TRIALS_PER_PROMPT_PER_FAULT_TYPE,
            "modes": len(MAIN_MODES),
            "fault_specs": TOTAL_FAULT_SPECS,
            "fault_runs": TOTAL_FAULT_RUNS,
            "controls": TOTAL_CONTROLS,
            "formula": "5*10*3*40*3=18000",
        }
        identity["main_modes"] = [
            {
                "mode_id": item.mode_id,
                "bounds_source": item.bounds_source.value,
                "correction": item.correction.value,
                "scaling_factor": item.scaling_factor,
            }
            for item in MAIN_MODES
        ]
        identity["protocol_corrections"] = {
            "primary_semantics": "paper_clamp",
            "out_of_bound": "saturate_to_lower_or_upper_bound",
            "nan": "replace_with_zero",
            "repository_cuda_conflict": (
                "submitted TensorCompare.cu zeros out-of-bound values"
            ),
            "batch_size": 1,
            "clean_state": "fresh bounds state for every inference",
            "qwen_attention": "sdpa",
            "opt_attention": "eager",
            "target_step_min": 1,
            "fixed_generation": {"qa": 60, "gsm8k": 180},
            "clean_failure_policy": (
                "retain runs but classify as NON_EVALUABLE_CLEAN_FAILURE"
            ),
        }
        source_paths = sorted(
            (
                REPO_ROOT / "reproduction" / "src" / "ft2_formal"
            ).glob("*.py")
        )
        source_paths.extend(
            [
                REPO_ROOT / "reproduction" / "run_pilot.py",
                REPO_ROOT / "reproduction" / "run_reduced_formal.py",
                REPO_ROOT / "reproduction" / "run_main_18k.py",
            ]
        )
        identity["source_files"] = _file_records(
            [path for path in source_paths if path.exists()]
        )
        dependency_paths = [
            self.formal_root / "campaign.json",
            self.formal_root / "main_status.json",
            self.formal_root / "selection.json",
            self.validation_path,
        ]
        for model_key, dataset_key in PAIR_ORDER:
            identifier = pair_id(model_key, dataset_key)
            dependency_paths.append(
                self.formal_root / "manifests" / f"{identifier}.json"
            )
            dependency_paths.append(
                self.formal_root / "offline_bounds" / f"{identifier}.json"
            )
            for item in self.parent_selection["pairs"][identifier][
                "screening_artifacts"
            ]:
                path = Path(item["path"])
                if not path.is_absolute():
                    path = REPO_ROOT / path
                dependency_paths.append(path)
        identity["parent_dependencies"] = {
            "parent_campaign_fingerprint":
                self.parent_campaign["campaign_fingerprint"],
            "parent_status": self.parent_status["status"],
            "equivalence_validation_artifact_sha256":
                self.validation["artifact_sha256"],
            "files": _file_records(dependency_paths),
        }
        return identity

    def _screen_path(
        self,
        model_key: str,
        dataset_key: str,
        source_position: int,
    ) -> Path:
        return (
            self.formal_root
            / "screening"
            / model_key
            / dataset_key
            / f"{source_position:02d}.json"
        )

    def _load_screen_record(
        self,
        model_key: str,
        dataset_key: str,
        source_position: int,
    ) -> dict[str, Any]:
        record = load_artifact(
            self._screen_path(model_key, dataset_key, source_position)
        )
        if (
            record.get("campaign_fingerprint")
            != self.parent_campaign["campaign_fingerprint"]
        ):
            raise RuntimeError("Parent screening fingerprint mismatch")
        return record

    def ensure_selection(self) -> dict[str, Any]:
        path = self.output_root / "selection.json"
        if path.exists():
            selection = load_artifact(path)
            self._assert_artifact_campaign(selection)
            self._validate_selection(selection)
            return selection
        payload = {
            key: value
            for key, value in self.parent_selection.items()
            if key not in {"artifact_sha256", "campaign_fingerprint"}
        }
        payload["campaign_fingerprint"] = self.campaign_fingerprint
        payload["imported_dependency"] = {
            "role": "frozen_formal_prompt_selection",
            "path": _relative(self.formal_root / "selection.json"),
            "artifact_sha256":
                self.parent_selection["artifact_sha256"],
            "producer_campaign_fingerprint":
                self.parent_campaign["campaign_fingerprint"],
        }
        payload["main18k_role"] = (
            "same ten prompts per pair; fresh 40-trial manifests"
        )
        selection = atomic_write_artifact(path, payload)
        self._validate_selection(selection)
        return selection

    def ensure_offline_bounds(
        self,
        model_key: str,
        dataset_key: str,
        model: Any,
        tokenizer: Any,
        adapter: ModelAdapter,
    ) -> tuple[dict[str, Bounds], dict[str, Any]]:
        del model, tokenizer
        identifier = pair_id(model_key, dataset_key)
        path = (
            self.formal_root / "offline_bounds" / f"{identifier}.json"
        )
        artifact = load_artifact(path)
        if (
            artifact.get("campaign_fingerprint")
            != self.parent_campaign["campaign_fingerprint"]
        ):
            raise RuntimeError("Parent offline-bounds fingerprint mismatch")
        bounds = {
            key: Bounds.from_dict(value)
            for key, value in artifact["bounds"].items()
        }
        expected = {site.key for site in adapter.critical_sites}
        if set(bounds) != expected:
            raise RuntimeError("Offline bounds key set is incomplete")
        if int(artifact["selection"]["count"]) != 200:
            raise RuntimeError("Offline calibration sample count changed")
        if (
            artifact["dataset_split"]
            == DATASETS[dataset_key].evaluation_split
        ):
            raise RuntimeError("Offline/evaluation splits overlap")
        return bounds, artifact

    def ensure_manifest(
        self,
        model_key: str,
        dataset_key: str,
        adapter: ModelAdapter,
        selection: Mapping[str, Any],
    ) -> tuple[FaultSpec, ...]:
        identifier = pair_id(model_key, dataset_key)
        entry = selection["pairs"][identifier]
        dataset_spec = DATASETS[dataset_key]
        expected = build_pair_manifest(
            adapter=adapter,
            dataset_key=dataset_key,
            dataset_indices=entry["dataset_indices"],
            target_steps=entry["target_steps"],
            trials_per_fault_type=
                TRIALS_PER_PROMPT_PER_FAULT_TYPE,
            generation_steps=dataset_spec.generation_steps,
            campaign_seed=196,
            max_input_tokens=1024,
        )
        if len(expected) != SPECS_PER_PAIR:
            raise AssertionError("A main pair must have 1,200 FaultSpecs")
        if len({spec.spec_id for spec in expected}) != SPECS_PER_PAIR:
            raise AssertionError("Main FaultSpec IDs are not unique")
        if any(spec.target_step < 1 for spec in expected):
            raise AssertionError("Main manifest contains a step-zero fault")
        parent_path = (
            self.formal_root / "manifests" / f"{identifier}.json"
        )
        _, parent_specs = load_manifest(parent_path)
        prefix_specs = tuple(
            spec for spec in expected if spec.trial_index < 3
        )
        if (
            len(parent_specs) != 90
            or manifest_sha256(prefix_specs)
            != manifest_sha256(parent_specs)
        ):
            raise RuntimeError(
                "The 40-trial manifest does not preserve frozen trials 0..2"
            )
        path = self._manifest_path(model_key, dataset_key)
        if path.exists():
            metadata, existing = load_manifest(path)
            if (
                metadata.get("campaign_fingerprint")
                != self.campaign_fingerprint
                or manifest_sha256(existing)
                != manifest_sha256(expected)
            ):
                raise RuntimeError("Existing main manifest changed")
            return existing
        metadata = {
            "campaign_fingerprint": self.campaign_fingerprint,
            "pair_id": identifier,
            "model_key": model_key,
            "model_revision": MODELS[model_key].revision,
            "dataset_key": dataset_key,
            "dataset_revision": dataset_spec.revision,
            "source_positions": entry["source_positions"],
            "dataset_indices": entry["dataset_indices"],
            "target_steps": entry["target_steps"],
            "author_sources":
                author_source_metadata(model_key, dataset_key),
            "campaign_seed": 196,
            "trials_per_prompt_per_fault_type":
                TRIALS_PER_PROMPT_PER_FAULT_TYPE,
            "derived_seed_algorithm":
                "SHA256 first 128 bits -> numpy.PCG64",
            "parent_trial_0_2_manifest": {
                "path": _relative(parent_path),
                "sha256": sha256_file(parent_path),
                "prefix_equivalence_passed": True,
            },
            "candidate_sites": [
                {
                    "key": site.key,
                    "module_path": site.module_path,
                    "critical": site.critical,
                    "sampling_weight": site.sampling_weight,
                    "out_features": site.out_features,
                }
                for site in adapter.sites
            ],
        }
        save_manifest(path, expected, metadata)
        print(
            f"[main18k-manifest] {identifier}: {len(expected)} specs",
            flush=True,
        )
        return expected

    def _model_fault_stage_complete(self, model_key: str) -> bool:
        for dataset_key in MODEL_DATASETS[model_key]:
            path = self._manifest_path(model_key, dataset_key)
            if not path.exists():
                return False
            try:
                _, specs = load_manifest(path)
            except Exception:
                return False
            if len(specs) != SPECS_PER_PAIR:
                return False
            for spec in specs:
                for protection in MAIN_MODES:
                    run_path = self._run_path(
                        model_key,
                        dataset_key,
                        spec.spec_id,
                        protection.mode_id,
                    )
                    if not run_path.exists():
                        return False
                    try:
                        self._validate_existing_run(
                            run_path, spec, protection.mode_id
                        )
                    except Exception:
                        return False
        return True

    def prepare(self) -> dict[str, Any]:
        path = self.output_root / "preparation.json"
        if path.exists():
            record = load_artifact(path)
            self._assert_artifact_campaign(record)
            return record
        selection = self.ensure_selection()
        for model_key in ("opt_2_7b", "qwen2_math_7b"):
            with self.loaded_model(model_key) as (
                model,
                tokenizer,
                adapter,
            ):
                self.ensure_model_preflight(
                    model_key, model, tokenizer, adapter
                )
                for dataset_key in MODEL_DATASETS[model_key]:
                    self.ensure_offline_bounds(
                        model_key,
                        dataset_key,
                        model,
                        tokenizer,
                        adapter,
                    )
                    self.ensure_manifest(
                        model_key,
                        dataset_key,
                        adapter,
                        selection,
                    )
            gc.collect()
        manifest_paths = sorted(
            (self.output_root / "manifests").glob("*.json")
        )
        if len(manifest_paths) != 5:
            raise AssertionError("Main preparation did not create five manifests")
        record = {
            "schema_version": 1,
            "campaign_fingerprint": self.campaign_fingerprint,
            "status": "prepared",
            "counts": {
                "pairs": 5,
                "prompts": 50,
                "fault_specs": TOTAL_FAULT_SPECS,
                "planned_fault_runs": TOTAL_FAULT_RUNS,
                "planned_controls": TOTAL_CONTROLS,
                "manifest_files": len(manifest_paths),
            },
            "selection_artifact_sha256":
                selection["artifact_sha256"],
            "manifest_files": [
                {
                    "path": _relative(item),
                    "sha256": sha256_file(item),
                }
                for item in manifest_paths
            ],
            "completed_at": utc_now(),
        }
        print("[main18k-prepare] PASS", flush=True)
        return atomic_write_artifact(path, record)

    def run(self) -> dict[str, Any]:
        self.prepare()
        selection = self.ensure_selection()
        for model_key in ("opt_2_7b", "qwen2_math_7b"):
            if self._model_fault_stage_complete(model_key):
                print(
                    f"[main18k] {model_key} already complete",
                    flush=True,
                )
                continue
            with self.loaded_model(model_key) as (
                model,
                tokenizer,
                adapter,
            ):
                self.ensure_model_preflight(
                    model_key, model, tokenizer, adapter
                )
                self.run_model_faults(
                    model_key,
                    model,
                    tokenizer,
                    adapter,
                    selection,
                )
            gc.collect()
        return audit_main18k(self)


def audit_main18k(campaign: Main18kCampaign) -> dict[str, Any]:
    root = campaign.output_root
    existing_audit = root / "audit.json"
    if existing_audit.exists():
        audit = load_artifact(existing_audit)
        campaign._assert_artifact_campaign(audit)
        if (
            audit.get("status") == "passed"
            and audit.get("counts", {}).get("fault_runs")
            == TOTAL_FAULT_RUNS
        ):
            return audit
    selection = campaign.ensure_selection()
    total_controls = 0
    total_runs = 0
    control_clean = Counter()
    outcome_counts = Counter()
    table_counts: dict[tuple[str, str, str], Counter] = defaultdict(Counter)

    for model_key, dataset_key in PAIR_ORDER:
        identifier = pair_id(model_key, dataset_key)
        metadata, specs = load_manifest(
            campaign._manifest_path(model_key, dataset_key)
        )
        if (
            metadata.get("campaign_fingerprint")
            != campaign.campaign_fingerprint
            or len(specs) != SPECS_PER_PAIR
            or len({spec.spec_id for spec in specs}) != SPECS_PER_PAIR
        ):
            raise RuntimeError(f"Invalid main manifest: {identifier}")
        entry = selection["pairs"][identifier]
        critical_count = 96 if model_key == "opt_2_7b" else 112
        generation_steps = DATASETS[dataset_key].generation_steps

        expected_control_names = {
            f"{position:02d}__{mode.mode_id}.json"
            for position in entry["source_positions"]
            for mode in MAIN_MODES
        }
        control_dir = root / "controls" / identifier
        observed_controls = {
            path.name for path in control_dir.glob("*.json")
        }
        if observed_controls != expected_control_names:
            raise RuntimeError(
                f"Control artifact set mismatch: {identifier}"
            )
        for position in entry["source_positions"]:
            for mode in MAIN_MODES:
                control = load_artifact(
                    campaign._control_path(
                        model_key,
                        dataset_key,
                        position,
                        mode.mode_id,
                    )
                )
                campaign._assert_artifact_campaign(control)
                if (
                    control.get("status") != "completed"
                    or control["engine"]["injection_count"] != 0
                    or len(control["output"]["token_ids"])
                    != generation_steps
                ):
                    raise RuntimeError("Invalid clean control")
                if (
                    mode.bounds_source is BoundsSource.NONE
                    and control["engine"]["correction_elements"] != 0
                ):
                    raise RuntimeError(
                        "No-protection control performed correction"
                    )
                if (
                    mode.bounds_source is BoundsSource.FIRST_TOKEN
                    and control["engine"]["online_bounds_key_count"]
                    != critical_count
                ):
                    raise RuntimeError(
                        "First-token control bounds are incomplete"
                    )
                control_clean[
                    (
                        identifier,
                        mode.mode_id,
                        bool(control["score"]["task_correct"]),
                    )
                ] += 1
                total_controls += 1

        expected_run_names = {
            f"{spec.spec_id}__{mode.mode_id}.json"
            for spec in specs
            for mode in MAIN_MODES
        }
        run_dir = root / "runs" / identifier
        observed_runs = {path.name for path in run_dir.glob("*.json")}
        if observed_runs != expected_run_names:
            raise RuntimeError(f"Run artifact set mismatch: {identifier}")

        for spec in specs:
            for mode in MAIN_MODES:
                path = campaign._run_path(
                    model_key,
                    dataset_key,
                    spec.spec_id,
                    mode.mode_id,
                )
                record = campaign._validate_existing_run(
                    path, spec, mode.mode_id
                )
                if record["terminal_status"] != "completed":
                    raise RuntimeError(
                        "Main target_step>=1 run did not complete"
                    )
                if len(record["output"]["token_ids"]) != generation_steps:
                    raise RuntimeError("Fixed generation length changed")
                engine = record["engine"]
                trace = engine["injection_trace"]
                if (
                    engine["injection_count"] != 1
                    or trace is None
                    or trace["spec_id"] != spec.spec_id
                    or trace["observed_step"] != spec.target_step
                    or trace["site_key"] != spec.site_key
                    or trace["flat_index"] != spec.flat_index
                    or trace["bit_positions"] != list(spec.bit_positions)
                    or not trace["bit_flip_verified"]
                    or trace["hamming_distance"]
                    != len(spec.bit_positions)
                ):
                    raise RuntimeError("Fault trace audit failed")
                before = int(trace["before_bits_hex"], 16)
                after = int(trace["after_bits_hex"], 16)
                mask = sum(1 << bit for bit in spec.bit_positions)
                if after != (before ^ mask):
                    raise RuntimeError("Raw FP16 XOR audit failed")
                if (
                    mode.bounds_source is BoundsSource.NONE
                    and engine["correction_elements"] != 0
                ):
                    raise RuntimeError(
                        "No-protection run corrected activations"
                    )
                if (
                    mode.bounds_source is BoundsSource.FIRST_TOKEN
                    and engine["online_bounds_key_count"]
                    != critical_count
                ):
                    raise RuntimeError(
                        "First-token run bounds are incomplete"
                    )
                classification = record["classification"]
                outcome = classification["outcome"]
                if outcome not in ACCEPTED_OUTCOMES:
                    raise RuntimeError(f"Unknown outcome: {outcome}")
                evaluable = bool(classification["evaluable"])
                if outcome == "NON_EVALUABLE_CLEAN_FAILURE":
                    if (
                        evaluable
                        or classification["masked"]
                        or classification["sdc"]
                    ):
                        raise RuntimeError(
                            "Clean failure entered SDC/masked denominator"
                        )
                elif not evaluable:
                    raise RuntimeError(
                        "An evaluable outcome was marked non-evaluable"
                    )
                outcome_counts[outcome] += 1
                table_counts[
                    (
                        identifier,
                        mode.mode_id,
                        spec.fault_type.value,
                    )
                ][outcome] += 1
                total_runs += 1

    if total_controls != TOTAL_CONTROLS:
        raise AssertionError(
            f"Expected {TOTAL_CONTROLS} controls, got {total_controls}"
        )
    if total_runs != TOTAL_FAULT_RUNS:
        raise AssertionError(
            f"Expected {TOTAL_FAULT_RUNS} runs, got {total_runs}"
        )

    tables: dict[str, Any] = {}
    for (identifier, mode_id, fault_type), counts in sorted(
        table_counts.items()
    ):
        cell = (
            tables
            .setdefault(identifier, {})
            .setdefault(mode_id, {})
        )
        evaluable = sum(
            count
            for outcome, count in counts.items()
            if outcome != "NON_EVALUABLE_CLEAN_FAILURE"
        )
        sdc = counts["SDC"]
        cell[fault_type] = {
            "planned": 400,
            "evaluable": evaluable,
            "outcome_counts": dict(sorted(counts.items())),
            "sdc": sdc,
            "sdc_rate_evaluable":
                sdc / evaluable if evaluable else None,
        }

    clean_summary: dict[str, Any] = {}
    for (identifier, mode_id, correct), count in sorted(
        control_clean.items()
    ):
        target = (
            clean_summary
            .setdefault(identifier, {})
            .setdefault(
                mode_id,
                {"clean_correct": 0, "clean_incorrect": 0},
            )
        )
        target[
            "clean_correct" if correct else "clean_incorrect"
        ] += count

    audit_payload = {
        "schema_version": 1,
        "campaign_fingerprint": campaign.campaign_fingerprint,
        "status": "passed",
        "counts": {
            "pairs": 5,
            "prompts": 50,
            "fault_specs": TOTAL_FAULT_SPECS,
            "controls": total_controls,
            "fault_runs": total_runs,
        },
        "invariants": {
            "one_campaign_fingerprint": True,
            "five_manifests_of_1200_specs": True,
            "exactly_three_modes_per_spec": True,
            "exactly_one_verified_xor_per_run": True,
            "hamming_distance_matches_fault_type": True,
            "target_step_minimum_is_one": True,
            "fixed_generation_lengths": True,
            "first_token_bounds_complete": True,
            "no_protection_has_zero_corrections": True,
            "clean_failures_excluded_from_sdc_denominator": True,
            "trial_0_2_matches_frozen_parent": True,
        },
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "audited_at": utc_now(),
    }
    audit = atomic_write_artifact(existing_audit, audit_payload)
    summary = atomic_write_artifact(
        root / "summary.json",
        {
            "schema_version": 1,
            "campaign_fingerprint": campaign.campaign_fingerprint,
            "status": "completed",
            "counts": audit["counts"],
            "clean_controls": clean_summary,
            "result_tables": tables,
            "outcome_counts": audit["outcome_counts"],
            "audit_artifact_sha256": audit["artifact_sha256"],
            "created_at": utc_now(),
        },
    )
    atomic_write_artifact(
        root / "completion.json",
        {
            "schema_version": 1,
            "campaign_fingerprint": campaign.campaign_fingerprint,
            "status": "completed",
            "counts": audit["counts"],
            "audit_artifact_sha256": audit["artifact_sha256"],
            "summary_artifact_sha256": summary["artifact_sha256"],
            "completed_at": utc_now(),
        },
    )
    print("[main18k-audit] PASS: 18,000/18,000", flush=True)
    return audit
