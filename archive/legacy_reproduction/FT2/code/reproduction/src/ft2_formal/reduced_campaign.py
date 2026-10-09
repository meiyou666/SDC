from __future__ import annotations

import gc
import traceback
from pathlib import Path
from typing import Any, Mapping

import torch

from .adapters import ModelAdapter
from .artifacts import (
    atomic_write_artifact,
    load_artifact,
    sha256_file,
    utc_now,
)
from .campaign import (
    CONFIG_PATH,
    DEFAULT_OUTPUT_ROOT,
    MAIN_MODES,
    PAIR_ORDER,
    PilotCampaign,
    _file_records,
    _input_metadata,
    _output_payload,
    _relative,
    pair_id,
)
from .decoding import fixed_greedy_generate, prepare_prompt
from .engine import FT2HookEngine
from .manifest import (
    build_pair_manifest,
    load_manifest,
    manifest_sha256,
    save_manifest,
)
from .schema import Bounds, BoundsSource
from .tasks import (
    DATASETS,
    MODELS,
    MODEL_DATASETS,
    REPO_ROOT,
    author_candidates,
    author_source_metadata,
    load_task_dataset,
    prompt_and_references,
    score_output,
)


PILOT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
FORMAL_OUTPUT_ROOT = (
    REPO_ROOT / "reproduction" / "results" / "formal_reduced_v1"
)
PROMPTS_PER_PAIR = 10
TRIALS_PER_PROMPT_PER_FAULT_TYPE = 3
SPECS_PER_PAIR = 90
TOTAL_FAULT_SPECS = 450
TOTAL_FAULT_RUNS = 1350


class ReducedFormalCampaign(PilotCampaign):
    """Frozen 4090 campaign with pilot-exposed prompts excluded."""

    def __init__(
        self,
        output_root: str | Path = FORMAL_OUTPUT_ROOT,
        *,
        pilot_root: str | Path = PILOT_OUTPUT_ROOT,
    ):
        self.pilot_root = Path(pilot_root).resolve()
        self.pilot_campaign = load_artifact(
            self.pilot_root / "campaign.json"
        )
        self.pilot_audit = load_artifact(self.pilot_root / "audit.json")
        self.pilot_fingerprint = self.pilot_campaign[
            "campaign_fingerprint"
        ]
        if (
            self.pilot_audit.get("status") != "passed"
            or self.pilot_audit.get("campaign_fingerprint")
            != self.pilot_fingerprint
        ):
            raise RuntimeError(
                "Pilot dependency is missing a matching passed audit"
            )
        super().__init__(output_root)

    def _validate_frozen_config(self) -> None:
        super()._validate_frozen_config()
        formal = self.config["reduced_formal"]
        expected = {
            "model_dataset_pairs": 5,
            "prompts_per_pair": PROMPTS_PER_PAIR,
            "faults_per_prompt_per_fault_type":
                TRIALS_PER_PROMPT_PER_FAULT_TYPE,
            "fault_types": 3,
            "main_protection_modes": 3,
            "matrix_cells": 45,
            "fault_inferences_per_cell": 30,
            "expected_fault_inferences": TOTAL_FAULT_RUNS,
        }
        for key, value in expected.items():
            if int(formal[key]) != value:
                raise ValueError(
                    f"Frozen reduced-formal setting changed: {key}"
                )
        selection = self.config["sample_selection"]
        if (
            int(selection["formal_prompts_per_model_dataset"])
            != PROMPTS_PER_PAIR
            or not bool(selection["exclude_pilot_prompts"])
            or int(selection["main_fault_target_step_min"]) != 1
        ):
            raise ValueError("Frozen formal selection policy changed")

    def _immutable_identity(self) -> dict[str, Any]:
        identity = super()._immutable_identity()
        identity["schema_version"] = 2
        identity["experiment_id"] = "ft2_rtx4090_reduced_formal_v1"
        identity.pop("pilot_dimensions", None)
        identity["reduced_formal_dimensions"] = {
            "pairs": len(PAIR_ORDER),
            "prompts_per_pair": PROMPTS_PER_PAIR,
            "trials_per_prompt_per_fault_type":
                TRIALS_PER_PROMPT_PER_FAULT_TYPE,
            "fault_types": 3,
            "modes": len(MAIN_MODES),
            "fault_specs": TOTAL_FAULT_SPECS,
            "fault_runs": TOTAL_FAULT_RUNS,
            "offline_examples_per_pair": 200,
        }
        identity["selection_policy"] = {
            "candidate_order": "author qid order",
            "eligibility": (
                "clean-correct, target_step>=1, not used by pilot faults"
            ),
            "qa": "first 10 positions eligible for both OPT and Qwen",
            "gsm8k": "first 10 eligible Qwen positions",
            "conditioning": "results condition on clean-correct prompts",
            "no_result_dependent_stopping": True,
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
            ]
        )
        identity["source_files"] = _file_records(
            [path for path in source_paths if path.exists()]
        )
        dependency_paths = [
            self.pilot_root / "campaign.json",
            self.pilot_root / "audit.json",
            self.pilot_root / "selection.json",
        ]
        dependency_paths.extend(
            sorted((self.pilot_root / "offline_bounds").glob("*.json"))
        )
        dependency_paths.extend(
            sorted((self.pilot_root / "screening").glob("*/*/*.json"))
        )
        identity["pilot_dependency_bundle"] = {
            "role": "clean_screening_and_offline_bounds",
            "producer_campaign_fingerprint": self.pilot_fingerprint,
            "audit_status": self.pilot_audit["status"],
            "audit_artifact_sha256":
                self.pilot_audit["artifact_sha256"],
            "auditor": self.pilot_audit.get("auditor"),
            "files": _file_records(dependency_paths),
        }
        identity["offline_reuse_precondition"] = (
            "calibration split must differ from evaluation split"
        )
        return identity

    def _pilot_selection(self) -> dict[str, Any]:
        selection = load_artifact(self.pilot_root / "selection.json")
        if selection.get("campaign_fingerprint") != self.pilot_fingerprint:
            raise RuntimeError("Pilot selection fingerprint mismatch")
        return selection

    def _pilot_screen_path(
        self,
        model_key: str,
        dataset_key: str,
        source_position: int,
    ) -> Path:
        return (
            self.pilot_root
            / "screening"
            / model_key
            / dataset_key
            / f"{source_position:02d}.json"
        )

    def _find_screen_record(
        self,
        model_key: str,
        dataset_key: str,
        source_position: int,
    ) -> tuple[dict[str, Any], Path] | None:
        formal_path = self._screen_path(
            model_key, dataset_key, source_position
        )
        if formal_path.exists():
            record = load_artifact(formal_path)
            self._assert_artifact_campaign(record)
            return record, formal_path
        pilot_path = self._pilot_screen_path(
            model_key, dataset_key, source_position
        )
        if pilot_path.exists():
            record = load_artifact(pilot_path)
            if (
                record.get("campaign_fingerprint")
                != self.pilot_fingerprint
            ):
                raise RuntimeError("Pilot screening fingerprint mismatch")
            return record, pilot_path
        return None

    def _screen_candidate(
        self,
        *,
        model_key: str,
        dataset_key: str,
        source_position: int,
        model: Any,
        tokenizer: Any,
        engine: FT2HookEngine,
    ) -> dict[str, Any]:
        candidate = author_candidates(model_key, dataset_key)[
            source_position
        ]
        dataset_spec = DATASETS[dataset_key]
        dataset = load_task_dataset(dataset_key)
        split = dataset_spec.evaluation_split
        example = dataset[split][candidate.dataset_index]
        prompt, references = prompt_and_references(dataset_key, example)
        prepared = prepare_prompt(
            tokenizer,
            prompt,
            device="cuda",
            max_input_tokens=1024,
        )
        protection = MAIN_MODES[0]
        started = utc_now()
        engine.start_inference(fault=None, protection=protection)
        try:
            output = fixed_greedy_generate(
                model,
                tokenizer,
                prepared,
                engine,
                num_new_tokens=dataset_spec.generation_steps,
            )
        except Exception as exc:
            partial = engine.abort()
            invalid = {
                "schema_version": 1,
                "campaign_fingerprint": self.campaign_fingerprint,
                "status": "invalid_clean",
                "model_key": model_key,
                "dataset_key": dataset_key,
                "source_position": source_position,
                "dataset_index": candidate.dataset_index,
                "target_step": candidate.target_step,
                "started_at": started,
                "ended_at": utc_now(),
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                "engine": partial.to_dict(),
            }
            atomic_write_artifact(
                self._screen_path(
                    model_key, dataset_key, source_position
                ),
                invalid,
            )
            raise RuntimeError(
                f"Formal clean screening failed at {source_position}"
            ) from exc
        score = score_output(dataset_key, output.text, references)
        record = {
            "schema_version": 1,
            "campaign_fingerprint": self.campaign_fingerprint,
            "status": "completed",
            "model_key": model_key,
            "model_revision": MODELS[model_key].revision,
            "dataset_key": dataset_key,
            "dataset_revision": dataset_spec.revision,
            "dataset_split": split,
            "dataset_fingerprint": dataset[split]._fingerprint,
            "source_position": source_position,
            "source_line": source_position + 1,
            "dataset_index": candidate.dataset_index,
            "target_step": candidate.target_step,
            "input": _input_metadata(tokenizer, prompt, prepared),
            "references": list(references),
            "output": _output_payload(output),
            "score": score,
            "engine": output.engine_record.to_dict(),
            "selection_role": "formal_clean_eligibility_screen",
            "started_at": started,
            "ended_at": utc_now(),
        }
        return atomic_write_artifact(
            self._screen_path(
                model_key, dataset_key, source_position
            ),
            record,
        )

    def ensure_gsm_screening(
        self,
        model: Any,
        tokenizer: Any,
        adapter: ModelAdapter,
    ) -> None:
        model_key = "qwen2_math_7b"
        dataset_key = "gsm8k"
        pilot_entry = self._pilot_selection()["pairs"][
            pair_id(model_key, dataset_key)
        ]
        exposed = set(pilot_entry["source_positions"])
        eligible: list[int] = []
        with FT2HookEngine(adapter) as engine:
            for candidate in author_candidates(model_key, dataset_key):
                found = self._find_screen_record(
                    model_key,
                    dataset_key,
                    candidate.source_position,
                )
                if found is None:
                    record = self._screen_candidate(
                        model_key=model_key,
                        dataset_key=dataset_key,
                        source_position=candidate.source_position,
                        model=model,
                        tokenizer=tokenizer,
                        engine=engine,
                    )
                else:
                    record, _ = found
                qualifies = (
                    record.get("status") == "completed"
                    and bool(record["score"]["task_correct"])
                    and candidate.target_step >= 1
                    and candidate.source_position not in exposed
                )
                if qualifies:
                    eligible.append(candidate.source_position)
                print(
                    "[formal-screen] qwen2_math_7b/gsm8k "
                    f"{candidate.source_position + 1}/50 "
                    f"new_eligible={len(eligible)}/{PROMPTS_PER_PAIR}",
                    flush=True,
                )
                if len(eligible) == PROMPTS_PER_PAIR:
                    return
        raise RuntimeError(
            "Fewer than 10 unexposed clean-correct GSM8K prompts "
            "with target_step>=1 in the 50 author candidates"
        )

    def _dependency_descriptor(
        self,
        *,
        role: str,
        source_path: Path,
        source: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "role": role,
            "producer_campaign_fingerprint": self.pilot_fingerprint,
            "path": _relative(source_path),
            "size": source_path.stat().st_size,
            "artifact_sha256": source["artifact_sha256"],
            "schema_version": source.get("schema_version"),
            "producer_config_sha256":
                self.pilot_campaign["immutable"]["config_sha256"],
            "producer_audit_artifact_sha256":
                self.pilot_audit["artifact_sha256"],
            "producer_auditor": self.pilot_audit.get("auditor"),
        }

    def _materialize_screen_reference(
        self,
        model_key: str,
        dataset_key: str,
        source_position: int,
    ) -> dict[str, Any]:
        destination = self._screen_path(
            model_key, dataset_key, source_position
        )
        if destination.exists():
            record = load_artifact(destination)
            self._assert_artifact_campaign(record)
            return record
        found = self._find_screen_record(
            model_key, dataset_key, source_position
        )
        if found is None:
            raise FileNotFoundError(
                f"Missing clean screen {model_key}/{dataset_key}/"
                f"{source_position}"
            )
        source, source_path = found
        if source_path == destination:
            return source
        if (
            source.get("status") != "completed"
            or not source["score"]["task_correct"]
        ):
            raise RuntimeError("Cannot import an ineligible clean screen")
        payload = {
            key: value
            for key, value in source.items()
            if key not in {"artifact_sha256", "campaign_fingerprint"}
        }
        payload["campaign_fingerprint"] = self.campaign_fingerprint
        payload["imported_dependency"] = self._dependency_descriptor(
            role="clean_screening",
            source_path=source_path,
            source=source,
        )
        payload["selection_role"] = (
            "formal_clean_eligibility_screen_reused_without_fault_results"
        )
        return atomic_write_artifact(destination, payload)

    def ensure_selection(self) -> dict[str, Any]:
        path = self._selection_path()
        if path.exists():
            selection = load_artifact(path)
            self._assert_artifact_campaign(selection)
            self._validate_selection(selection)
            return selection

        pilot_selection = self._pilot_selection()
        traces: dict[str, list[dict[str, Any]]] = {}
        selected_positions: dict[str, list[int]] = {}

        for dataset_key in ("squad_v2", "xtreme_mlqa_en_en"):
            trace: list[dict[str, Any]] = []
            chosen: list[int] = []
            opt_exposed = set(
                pilot_selection["pairs"][
                    pair_id("opt_2_7b", dataset_key)
                ]["source_positions"]
            )
            qwen_exposed = set(
                pilot_selection["pairs"][
                    pair_id("qwen2_math_7b", dataset_key)
                ]["source_positions"]
            )
            opt_candidates = author_candidates(
                "opt_2_7b", dataset_key
            )
            qwen_candidates = author_candidates(
                "qwen2_math_7b", dataset_key
            )
            for source_position in range(50):
                opt_found = self._find_screen_record(
                    "opt_2_7b", dataset_key, source_position
                )
                qwen_found = self._find_screen_record(
                    "qwen2_math_7b", dataset_key, source_position
                )
                if opt_found is None or qwen_found is None:
                    raise RuntimeError(
                        f"Missing audited QA screen at {source_position}"
                    )
                opt_record, opt_path = opt_found
                qwen_record, qwen_path = qwen_found
                not_exposed = (
                    source_position not in opt_exposed
                    and source_position not in qwen_exposed
                )
                valid_steps = (
                    opt_candidates[source_position].target_step >= 1
                    and qwen_candidates[source_position].target_step >= 1
                )
                clean_correct = (
                    opt_record.get("status") == "completed"
                    and qwen_record.get("status") == "completed"
                    and bool(opt_record["score"]["task_correct"])
                    and bool(qwen_record["score"]["task_correct"])
                )
                same_example = (
                    opt_candidates[source_position].dataset_index
                    == qwen_candidates[source_position].dataset_index
                )
                qualifies = (
                    not_exposed
                    and valid_steps
                    and clean_correct
                    and same_example
                )
                trace.append(
                    {
                        "source_position": source_position,
                        "qualifies": qualifies,
                        "not_pilot_exposed": not_exposed,
                        "target_steps_ge_1": valid_steps,
                        "both_clean_correct": clean_correct,
                        "same_dataset_index": same_example,
                        "opt_screen_path": _relative(opt_path),
                        "opt_screen_artifact_sha256":
                            opt_record["artifact_sha256"],
                        "qwen_screen_path": _relative(qwen_path),
                        "qwen_screen_artifact_sha256":
                            qwen_record["artifact_sha256"],
                    }
                )
                if qualifies:
                    chosen.append(source_position)
                if len(chosen) == PROMPTS_PER_PAIR:
                    break
            if len(chosen) != PROMPTS_PER_PAIR:
                raise RuntimeError(
                    f"Fewer than 10 eligible shared QA prompts: "
                    f"{dataset_key}"
                )
            traces[dataset_key] = trace
            selected_positions[dataset_key] = chosen

        gsm_trace: list[dict[str, Any]] = []
        gsm_chosen: list[int] = []
        gsm_entry = pilot_selection["pairs"][
            pair_id("qwen2_math_7b", "gsm8k")
        ]
        gsm_exposed = set(gsm_entry["source_positions"])
        gsm_candidates = author_candidates("qwen2_math_7b", "gsm8k")
        for candidate in gsm_candidates:
            found = self._find_screen_record(
                "qwen2_math_7b",
                "gsm8k",
                candidate.source_position,
            )
            if found is None:
                raise RuntimeError(
                    "GSM8K screening must complete before selection"
                )
            record, source_path = found
            not_exposed = candidate.source_position not in gsm_exposed
            valid_step = candidate.target_step >= 1
            clean_correct = (
                record.get("status") == "completed"
                and bool(record["score"]["task_correct"])
            )
            qualifies = not_exposed and valid_step and clean_correct
            gsm_trace.append(
                {
                    "source_position": candidate.source_position,
                    "qualifies": qualifies,
                    "not_pilot_exposed": not_exposed,
                    "target_step_ge_1": valid_step,
                    "clean_correct": clean_correct,
                    "screen_path": _relative(source_path),
                    "screen_artifact_sha256":
                        record["artifact_sha256"],
                }
            )
            if qualifies:
                gsm_chosen.append(candidate.source_position)
            if len(gsm_chosen) == PROMPTS_PER_PAIR:
                break
        if len(gsm_chosen) != PROMPTS_PER_PAIR:
            raise RuntimeError(
                "Fewer than 10 eligible unexposed GSM8K prompts"
            )
        traces["gsm8k"] = gsm_trace
        selected_positions["gsm8k"] = gsm_chosen

        pairs: dict[str, Any] = {}
        for model_key, dataset_key in PAIR_ORDER:
            positions = selected_positions[dataset_key]
            candidates = author_candidates(model_key, dataset_key)
            rows = [candidates[position] for position in positions]
            screening_artifacts = []
            for position in positions:
                record = self._materialize_screen_reference(
                    model_key, dataset_key, position
                )
                screening_artifacts.append(
                    {
                        "path": _relative(
                            self._screen_path(
                                model_key, dataset_key, position
                            )
                        ),
                        "artifact_sha256": record["artifact_sha256"],
                    }
                )
            pairs[pair_id(model_key, dataset_key)] = {
                "model_key": model_key,
                "dataset_key": dataset_key,
                "source_positions": positions,
                "dataset_indices": [
                    row.dataset_index for row in rows
                ],
                "target_steps": [row.target_step for row in rows],
                "pilot_excluded_source_positions":
                    pilot_selection["pairs"][
                        pair_id(model_key, dataset_key)
                    ]["source_positions"],
                "screening_artifacts": screening_artifacts,
            }

        payload = {
            "schema_version": 1,
            "campaign_fingerprint": self.campaign_fingerprint,
            "policy": (
                "first 10 author-order clean-correct prompts with "
                "target_step>=1, excluding pilot fault prompts; "
                "QA positions shared between OPT and Qwen"
            ),
            "conditioning": (
                "inference is conditional on clean-correct prompts"
            ),
            "pilot_dependency_campaign_fingerprint":
                self.pilot_fingerprint,
            "pairs": pairs,
            "screening_trace": traces,
            "created_at": utc_now(),
        }
        selection = atomic_write_artifact(path, payload)
        self._validate_selection(selection)
        print(
            "[formal-selection] locked 10 unexposed prompts for all pairs",
            flush=True,
        )
        return selection

    def _validate_selection(
        self,
        selection: Mapping[str, Any],
    ) -> None:
        expected_ids = {
            pair_id(model_key, dataset_key)
            for model_key, dataset_key in PAIR_ORDER
        }
        pairs = selection.get("pairs", {})
        if set(pairs) != expected_ids:
            raise RuntimeError("Formal selection does not have five pairs")
        pilot_pairs = self._pilot_selection()["pairs"]
        for identifier, entry in pairs.items():
            fields = (
                "source_positions",
                "dataset_indices",
                "target_steps",
            )
            if any(
                len(entry[field]) != PROMPTS_PER_PAIR
                for field in fields
            ):
                raise RuntimeError(
                    f"{identifier} does not have 10 formal prompts"
                )
            if len(set(entry["source_positions"])) != PROMPTS_PER_PAIR:
                raise RuntimeError("Formal selection contains duplicates")
            if set(entry["source_positions"]).intersection(
                pilot_pairs[identifier]["source_positions"]
            ):
                raise RuntimeError(
                    "Formal selection contains a pilot-exposed prompt"
                )
            if any(int(step) < 1 for step in entry["target_steps"]):
                raise RuntimeError(
                    "Formal main selection contains target_step=0"
                )
            for source_position in entry["source_positions"]:
                record = self._load_screen_record(
                    entry["model_key"],
                    entry["dataset_key"],
                    source_position,
                )
                if (
                    record.get("status") != "completed"
                    or not record["score"]["task_correct"]
                ):
                    raise RuntimeError(
                        "Selected formal clean screen is not correct"
                    )
        for dataset_key in ("squad_v2", "xtreme_mlqa_en_en"):
            if (
                pairs[pair_id("opt_2_7b", dataset_key)][
                    "source_positions"
                ]
                != pairs[pair_id("qwen2_math_7b", dataset_key)][
                    "source_positions"
                ]
            ):
                raise RuntimeError(
                    "OPT/Qwen QA formal positions are not shared"
                )

    def ensure_offline_bounds(
        self,
        model_key: str,
        dataset_key: str,
        model: Any,
        tokenizer: Any,
        adapter: ModelAdapter,
    ) -> tuple[dict[str, Bounds], dict[str, Any]]:
        del model, tokenizer
        destination = self._bounds_path(model_key, dataset_key)
        if destination.exists():
            artifact = load_artifact(destination)
            self._assert_artifact_campaign(artifact)
        else:
            source_path = (
                self.pilot_root
                / "offline_bounds"
                / f"{pair_id(model_key, dataset_key)}.json"
            )
            source = load_artifact(source_path)
            if (
                source.get("campaign_fingerprint")
                != self.pilot_fingerprint
            ):
                raise RuntimeError("Pilot bounds fingerprint mismatch")
            if (
                source["dataset_split"]
                == DATASETS[dataset_key].evaluation_split
            ):
                raise RuntimeError(
                    "Offline calibration split overlaps evaluation split"
                )
            payload = {
                key: value
                for key, value in source.items()
                if key not in {
                    "artifact_sha256",
                    "campaign_fingerprint",
                }
            }
            payload["campaign_fingerprint"] = self.campaign_fingerprint
            payload["provenance"] = (
                "reused_from_audited_pilot_offline_reduced_200"
            )
            payload["imported_dependency"] = (
                self._dependency_descriptor(
                    role="offline_bounds",
                    source_path=source_path,
                    source=source,
                )
            )
            payload["evaluation_split_disjoint_check"] = {
                "calibration_split": source["dataset_split"],
                "evaluation_split":
                    DATASETS[dataset_key].evaluation_split,
                "passed": True,
            }
            artifact = atomic_write_artifact(destination, payload)

        bounds = {
            key: Bounds.from_dict(value)
            for key, value in artifact["bounds"].items()
        }
        expected_keys = {
            site.key for site in adapter.critical_sites
        }
        if set(bounds) != expected_keys:
            raise RuntimeError("Reused offline bounds keys are incomplete")
        if (
            artifact["selection"]["count"] != 200
            or len(artifact["selection"]["indices"]) != 200
        ):
            raise RuntimeError("Reused offline bounds sample count changed")
        return bounds, artifact

    def ensure_manifest(
        self,
        model_key: str,
        dataset_key: str,
        adapter: ModelAdapter,
        selection: Mapping[str, Any],
    ) -> tuple[Any, ...]:
        entry = selection["pairs"][pair_id(model_key, dataset_key)]
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
            raise AssertionError(
                "Formal pair manifest must have 90 specs"
            )
        if any(spec.target_step < 1 for spec in expected):
            raise AssertionError(
                "Formal main manifest includes a step-zero fault"
            )
        path = self._manifest_path(model_key, dataset_key)
        if path.exists():
            metadata, existing = load_manifest(path)
            if (
                metadata.get("campaign_fingerprint")
                != self.campaign_fingerprint
            ):
                raise RuntimeError(
                    "Formal manifest belongs to another campaign"
                )
            if manifest_sha256(existing) != manifest_sha256(expected):
                raise RuntimeError(
                    "Frozen formal manifest differs from regeneration"
                )
            return existing
        metadata = {
            "campaign_fingerprint": self.campaign_fingerprint,
            "pair_id": pair_id(model_key, dataset_key),
            "model_key": model_key,
            "model_revision": MODELS[model_key].revision,
            "dataset_key": dataset_key,
            "dataset_revision": dataset_spec.revision,
            "source_positions": entry["source_positions"],
            "dataset_indices": entry["dataset_indices"],
            "target_steps": entry["target_steps"],
            "pilot_excluded_source_positions":
                entry["pilot_excluded_source_positions"],
            "author_sources":
                author_source_metadata(model_key, dataset_key),
            "campaign_seed": 196,
            "trials_per_prompt_per_fault_type":
                TRIALS_PER_PROMPT_PER_FAULT_TYPE,
            "derived_seed_algorithm":
                "SHA256 first 128 bits -> numpy.PCG64",
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
            f"[formal-manifest] saved {pair_id(model_key, dataset_key)} "
            f"({len(expected)} specs)",
            flush=True,
        )
        return expected

    def _model_fault_stage_complete(self, model_key: str) -> bool:
        for dataset_key in MODEL_DATASETS[model_key]:
            if not self._bounds_path(model_key, dataset_key).exists():
                return False
            manifest_path = self._manifest_path(
                model_key, dataset_key
            )
            if not manifest_path.exists():
                return False
            try:
                _, specs = load_manifest(manifest_path)
            except Exception:
                return False
            if len(specs) != SPECS_PER_PAIR:
                return False
            for spec in specs:
                for protection in MAIN_MODES:
                    path = self._run_path(
                        model_key,
                        dataset_key,
                        spec.spec_id,
                        protection.mode_id,
                    )
                    if not path.exists():
                        return False
                    try:
                        self._validate_existing_run(
                            path, spec, protection.mode_id
                        )
                    except Exception:
                        return False
        return True

    def prepare(self) -> dict[str, Any]:
        preparation_path = self.output_root / "preparation.json"
        if preparation_path.exists():
            preparation = load_artifact(preparation_path)
            self._assert_artifact_campaign(preparation)
            return preparation

        with self.loaded_model("qwen2_math_7b") as (
            model,
            tokenizer,
            adapter,
        ):
            self.ensure_model_preflight(
                "qwen2_math_7b", model, tokenizer, adapter
            )
            if not self._selection_path().exists():
                self.ensure_gsm_screening(model, tokenizer, adapter)
            selection = self.ensure_selection()
            for dataset_key in MODEL_DATASETS["qwen2_math_7b"]:
                self.ensure_offline_bounds(
                    "qwen2_math_7b",
                    dataset_key,
                    model,
                    tokenizer,
                    adapter,
                )
                self.ensure_manifest(
                    "qwen2_math_7b",
                    dataset_key,
                    adapter,
                    selection,
                )
        gc.collect()
        torch.cuda.empty_cache()

        with self.loaded_model("opt_2_7b") as (
            model,
            tokenizer,
            adapter,
        ):
            self.ensure_model_preflight(
                "opt_2_7b", model, tokenizer, adapter
            )
            for dataset_key in MODEL_DATASETS["opt_2_7b"]:
                self.ensure_offline_bounds(
                    "opt_2_7b",
                    dataset_key,
                    model,
                    tokenizer,
                    adapter,
                )
                self.ensure_manifest(
                    "opt_2_7b",
                    dataset_key,
                    adapter,
                    selection,
                )
        gc.collect()
        torch.cuda.empty_cache()

        manifest_files = sorted(
            (self.output_root / "manifests").glob("*.json")
        )
        preparation = {
            "schema_version": 1,
            "campaign_fingerprint": self.campaign_fingerprint,
            "status": "prepared",
            "counts": {
                "pairs": len(PAIR_ORDER),
                "selected_prompts":
                    len(PAIR_ORDER) * PROMPTS_PER_PAIR,
                "fault_specs": TOTAL_FAULT_SPECS,
                "planned_fault_runs": TOTAL_FAULT_RUNS,
                "manifest_files": len(manifest_files),
            },
            "selection_artifact_sha256":
                selection["artifact_sha256"],
            "manifest_files": [
                {
                    "path": _relative(path),
                    "sha256": sha256_file(path),
                }
                for path in manifest_files
            ],
            "completed_at": utc_now(),
        }
        print("[formal-prepare] PASS", flush=True)
        return atomic_write_artifact(preparation_path, preparation)

    def run(self) -> dict[str, Any]:
        self.prepare()
        selection = self.ensure_selection()
        for model_key in ("qwen2_math_7b", "opt_2_7b"):
            if self._model_fault_stage_complete(model_key):
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
            torch.cuda.empty_cache()

        from .audit import audit_reduced_formal

        return audit_reduced_formal(
            self.output_root,
            campaign_fingerprint=self.campaign_fingerprint,
            write_outputs=True,
        )

