"""Full-parameter continued pretraining through the shared Transformers Trainer.

One call to :func:`train` performs a complete run: it resolves the device from
the container, loads the prepared data, builds the model in FP32 with BF16
autocast, trains for ``max_steps`` optimizer updates, evaluates, and writes

``config.resolved.json``
    the configuration actually used, after device resolution
``run.json``
    config signature, data digest, ``uv.lock`` hash, commit and device
``metrics.jsonl``
    one JSON object per Trainer log line
``result.json``
    final training and validation metrics, or the failure reason
``checkpoint-<step>/``
    model, optimizer, scheduler and RNG state for ``--resume``

``model_transform`` receives the loaded model and ``transform_params`` before
the Trainer is built. Its versioned ``transform_id`` and parameters are saved
and checked on resume; it is where protected GEMM layers are installed.
``callbacks`` takes ``TrainerCallback`` instances, which reach the live model
and step counter.
"""

from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import subprocess
from typing import Any
import uuid

from accelerate.utils import is_npu_available
import torch
from transformers import (
    AutoConfig, AutoModelForCausalLM, PreTrainedModel, Trainer, TrainerCallback, TrainingArguments,
)
from transformers.trainer_utils import enable_full_determinism, get_last_checkpoint, set_seed

from .config import ROOT, Config, file_digest, model_source, resolve_path, run_signature
from .data import CausalCollator, load_tokenizer, prepared_digest, read_prepared
from .metrics import JsonMetrics, NonFiniteLossError, check_finite_loss, write_json


#: ``device: accelerator`` resolves through the container's SDC_RUNTIME.
RUNTIME_DEVICES = {"nvidia": "cuda", "ascend": "npu"}

#: Files a checkpoint must contain before it is accepted for ``--resume``.
CHECKPOINT_FILES = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth")


def transform_record(
    transform: Callable[..., PreTrainedModel] | None,
    transform_id: str | None,
    parameters: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Freeze the transform identity and the JSON parameters passed to it."""
    if transform is None:
        if transform_id is not None or parameters is not None:
            raise ValueError("设置变换标识或参数时，必须同时提供 model_transform。")
        return None
    if not callable(transform) or not isinstance(transform_id, str) or not transform_id.strip():
        raise ValueError("model_transform 必须可调用，并提供非空的 transform_id。")
    if parameters is None:
        parameters = {}
    if not isinstance(parameters, dict) or any(not isinstance(key, str) for key in parameters):
        raise ValueError("transform_params 必须是字符串键组成的参数字典。")
    try:
        parameters = json.loads(json.dumps(parameters, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("transform_params 必须能保存为 JSON，且不能包含非有限值。") from exc
    return {"id": transform_id, "parameters": parameters}


class ResumeTrainer(Trainer):
    """Stop on incomplete weight loading, allowing aliases of restored tensors."""

    def _issue_warnings_after_load(self, load_result):
        missing = set(load_result.missing_keys)
        # Safetensors saves tied tensors once. An omitted alias is valid only
        # when another name for the very same parameter/buffer was restored.
        state_keys = set(self.model.state_dict())
        aliases: dict[int, set[str]] = {}
        for name, tensor in (
            *self.model.named_parameters(remove_duplicate=False),
            *self.model.named_buffers(remove_duplicate=False),
        ):
            if name in state_keys:
                aliases.setdefault(id(tensor), set()).add(name)
        for names in aliases.values():
            if names - missing:
                missing.difference_update(names)
        if missing or load_result.unexpected_keys:
            raise ValueError(
                "检查点权重与当前模型不一致："
                f"缺失 {sorted(missing)}；多余 {sorted(load_result.unexpected_keys)}。"
            )


class RunRecord(TrainerCallback):
    """Write run settings after checkpoint loading has succeeded."""

    def __init__(self, output: Path, config: Config, record: dict[str, Any]):
        self.output, self.config, self.record = output, config, record

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            write_json(self.output / "run.json", self.record)
            write_json(self.output / "config.resolved.json", self.config.to_dict())


def require_device(device: str) -> None:
    """Check the requested device before the model is loaded."""
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("容器中没有可用的 CUDA 设备，请使用维护好的 NVIDIA 容器。")
    if device == "npu" and not is_npu_available():
        raise ValueError("没有可用的 NPU：昇腾容器需要匹配的 CANN、torch_npu 和设备映射。")


def resolve_runtime_config(config: Config) -> Config:
    """Resolve ``device: accelerator`` and ``{device}`` from the container."""
    device = config.training.device
    if device == "accelerator":
        runtime = os.environ.get("SDC_RUNTIME", "")
        device = RUNTIME_DEVICES.get(runtime)
        if device is None:
            raise ValueError(
                f"Qwen 训练任务请在 nvidia 或 ascend 容器中运行，当前 SDC_RUNTIME={runtime!r}。"
            )
    return replace(config, training=replace(
        config.training, device=device,
        output_dir=config.training.output_dir.replace("{device}", device),
    ))


def training_arguments(config: Config) -> TrainingArguments:
    """Translate the config into Trainer arguments.

    FP32 master weights with BF16 autocast, TF32 off, attention kernel from the
    config.
    """
    t = config.training
    require_device(t.device)
    torch.set_num_threads(t.cpu_threads)
    args = TrainingArguments(
        output_dir=str(resolve_path(t.output_dir)),
        use_cpu=t.device == "cpu", bf16=t.precision == "bf16", fp16=t.precision == "fp16",
        tf32=False, seed=t.seed, data_seed=t.seed, full_determinism=t.full_determinism,
        max_steps=t.max_steps,
        per_device_train_batch_size=t.batch_size, per_device_eval_batch_size=t.eval_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        learning_rate=t.learning_rate, weight_decay=t.weight_decay, warmup_steps=t.warmup_steps,
        lr_scheduler_type="cosine", optim="adamw_torch", max_grad_norm=t.max_grad_norm,
        logging_strategy="steps", logging_steps=t.logging_steps, logging_first_step=True,
        logging_nan_inf_filter=False,
        eval_strategy="steps", eval_steps=t.eval_steps,
        save_strategy="steps", save_steps=t.save_steps, save_total_limit=t.save_total_limit,
        gradient_checkpointing=config.model.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=t.dataloader_num_workers, dataloader_pin_memory=t.device != "cpu",
        dataloader_persistent_workers=False, dataloader_drop_last=False,
        ddp_find_unused_parameters=False, report_to=[], disable_tqdm=True,
        prediction_loss_only=True, save_safetensors=True,
    )
    if args.device.type != t.device:
        raise ValueError(f"配置要求 {t.device}，但 Trainer 选择了 {args.device.type}。")
    if args.world_size > 1:
        raise ValueError("实验按单卡运行：请不要用 torchrun 或多进程启动。")
    if args.n_gpu > 1:
        raise ValueError("实验按单卡运行：请只向容器开放一张设备。")
    return args


def resolve_checkpoint(output: Path, resume: str | None) -> str | None:
    """Pick the checkpoint to resume from, or verify the output dir is unused."""
    if resume is None:
        if output.exists() and any(output.iterdir()):
            raise ValueError(
                f"output_dir 非空：{output}。续训请加 --resume latest；"
                "上次训练没有留下检查点时，删除该目录或换一个 output_dir 重新开始。"
            )
        return None
    if resume == "latest":
        checkpoint = get_last_checkpoint(str(output)) if output.is_dir() else None
        if checkpoint is None:
            raise ValueError(
                f"{output} 里没有检查点：删除该目录或换一个 output_dir 重新开始训练。"
            )
    else:
        checkpoint = str(resolve_path(resume))
    for name in CHECKPOINT_FILES:
        if not (Path(checkpoint) / name).is_file():
            raise ValueError(f"{checkpoint} 不是完整的检查点：缺少 {name}。")
    return checkpoint


def validate_resume(
    checkpoint: str, config: Config, data_hash: str, transform: dict[str, Any] | None,
) -> None:
    """Refuse to resume into different settings, data or software."""
    directory = Path(checkpoint)
    filename = directory.parent / "run.json"
    if not filename.is_file():
        raise ValueError("续训需要原始运行目录中的 run.json。")
    previous = json.loads(filename.read_text())
    if "model_transform" not in previous:
        raise ValueError("续训记录缺少 model_transform 信息，请使用新目录重新开始训练。")
    current = {
        "config_signature": run_signature(config), "data_digest": data_hash,
        "lock_sha256": file_digest(ROOT / "uv.lock"), "model_transform": transform,
    }
    for key, value in current.items():
        if previous.get(key) != value:
            raise ValueError(f"续训设置不一致：{key}。")
    state = json.loads((directory / "trainer_state.json").read_text())
    if state["global_step"] >= config.training.max_steps:
        raise ValueError("这个检查点已经达到 max_steps。")


def _git_commit() -> str | None:
    """Current commit, or None outside a git checkout."""
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={ROOT}", "rev-parse", "HEAD"],
            cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _run_record(
    config: Config, args: TrainingArguments, data_hash: str, checkpoint: str | None,
    transform: dict[str, Any] | None,
) -> dict[str, Any]:
    """Assemble the run.json provenance record.

    The settings themselves are in ``config.resolved.json``; this file records
    what the run cannot restate, so a resume can refuse a changed setup.
    """
    device = getattr(torch, args.device.type, None)
    device_name = device.get_device_name(args.device) if hasattr(device, "get_device_name") else platform.machine()
    return {
        "config_signature": run_signature(config), "data_digest": data_hash,
        "lock_sha256": file_digest(ROOT / "uv.lock"), "git_commit": _git_commit(),
        "device_name": device_name, "model_transform": transform,
        "started_at": datetime.now(timezone.utc).isoformat(), "resume_from": checkpoint,
    }


def _load_model(config: Config, offline: bool):
    """Load the pinned model in FP32 with the configured attention kernel."""
    source = model_source(config.model)
    model_config = AutoConfig.from_pretrained(
        source, revision=config.model.revision, trust_remote_code=False, local_files_only=offline,
    )
    limit = getattr(model_config, "max_position_embeddings", getattr(model_config, "n_positions", None))
    if limit is not None and config.data.block_size > limit:
        raise ValueError(f"block_size 超过模型上下文长度 {limit}。")
    model = AutoModelForCausalLM.from_pretrained(
        source, revision=config.model.revision, config=model_config,
        dtype=torch.float32, attn_implementation=config.model.attention,
        trust_remote_code=False, local_files_only=offline,
    )
    model.config.use_cache = False
    return model


def train(
    config: Config,
    resume: str | None = None,
    offline: bool = False,
    callbacks: Sequence[TrainerCallback] = (),
    model_transform: Callable[..., PreTrainedModel] | None = None,
    *,
    transform_id: str | None = None,
    transform_params: dict[str, Any] | None = None,
) -> tuple[Trainer, dict[str, Any]]:
    """Run one training job and return the Trainer and its final metrics."""
    transform = transform_record(model_transform, transform_id, transform_params)
    config = resolve_runtime_config(config)
    output = resolve_path(config.training.output_dir)
    checkpoint = resolve_checkpoint(output, resume)
    datasets, data_manifest = read_prepared(config)
    args = training_arguments(config)
    data_hash = prepared_digest(data_manifest)
    if checkpoint:
        validate_resume(checkpoint, config, data_hash, transform)
    session = uuid.uuid4().hex
    output.mkdir(parents=True, exist_ok=True)
    if config.training.full_determinism:
        enable_full_determinism(config.training.seed)
    else:
        set_seed(config.training.seed)
    tokenizer = load_tokenizer(config, offline)
    model = _load_model(config, offline)
    if model_transform is not None:
        # Pass a copy so a transform cannot mutate the saved parameter record.
        model = model_transform(model, **json.loads(json.dumps(transform["parameters"])))
    record = _run_record(config, args, data_hash, checkpoint, transform)
    trainer = ResumeTrainer(
        model=model, args=args, processing_class=tokenizer,
        train_dataset=datasets["train"], eval_dataset=datasets["validation"],
        data_collator=CausalCollator(),
        callbacks=[RunRecord(output, config, record), JsonMetrics(output, session), *callbacks],
    )
    try:
        result = trainer.train(resume_from_checkpoint=checkpoint)
        evaluation = trainer.evaluate(metric_key_prefix="validation")
        check_finite_loss({**result.metrics, **evaluation})
    except NonFiniteLossError as exc:
        write_json(output / "result.json", {
            "status": "failed", "reason": "nonfinite_loss", "message": str(exc),
            "global_step": trainer.state.global_step, "epoch": trainer.state.epoch,
            **exc.metrics,
        })
        raise
    except (Exception, KeyboardInterrupt) as exc:
        # Out of memory and interrupts leave a result.json too, so a stopped
        # run is distinguishable from one that never started.
        write_json(output / "result.json", {
            "status": "failed", "reason": type(exc).__name__, "message": str(exc),
            "global_step": trainer.state.global_step, "epoch": trainer.state.epoch,
        })
        raise
    loss = evaluation["validation_loss"]
    metrics = {
        **result.metrics, **evaluation, "status": "completed",
        "global_step": trainer.state.global_step, "epoch": trainer.state.epoch,
        "validation_perplexity": math.exp(loss) if math.isfinite(loss) and loss < 700 else None,
    }
    if checkpoint:
        # Trainer averages this call's loss over the total step count, which
        # includes the steps before the resume, so the value would be too low.
        metrics.pop("train_loss", None)
    write_json(output / "result.json", metrics)
    trainer.save_state()
    return trainer, metrics
