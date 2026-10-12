"""Full-parameter continued pretraining through the shared Transformers Trainer.

One call to :func:`train` performs a complete run: it resolves the device from
the container, loads the prepared data, builds the model in FP32 with BF16
autocast, trains for ``max_steps`` optimizer updates, evaluates, and writes

``config.resolved.json``
    the configuration actually used, after device resolution
``run.json``
    software versions, data digest, device and training budget
``metrics.jsonl``
    one JSON object per Trainer log line
``result.json``
    final training and validation metrics, or the failure reason
``checkpoint-<step>/``
    model, optimizer, scheduler and RNG state for ``--resume``

``callbacks`` takes ``TrainerCallback`` instances, which reach the live model
and step counter.
"""

from collections.abc import Sequence
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
from transformers import AutoConfig, AutoModelForCausalLM, Trainer, TrainerCallback, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from .config import ROOT, Config, file_digest, model_source, resolve_path, run_signature
from .data import CausalCollator, load_tokenizer, prepared_digest, read_prepared
from .metrics import (
    JsonMetrics, NonFiniteLossError, check_finite_loss, package_versions, write_json,
)


#: ``device: accelerator`` resolves through the container's SDC_RUNTIME.
RUNTIME_DEVICES = {"nvidia": "cuda", "ascend": "npu"}

#: Files a checkpoint must contain before it is accepted for ``--resume``.
CHECKPOINT_FILES = ("optimizer.pt", "scheduler.pt")


def require_device(device: str) -> None:
    """Check the requested device before the model is loaded."""
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("容器中没有可用的 CUDA 设备，请使用维护好的 NVIDIA 容器。")
    if device == "npu" and not is_npu_available():
        raise ValueError("没有可用的 NPU：昇腾容器需要匹配的 CANN、torch_npu 和设备映射。")


def resolve_runtime_config(config: Config) -> Config:
    """Resolve ``device: accelerator`` and ``{backend}`` from the container."""
    device = config.training.device
    if device == "accelerator":
        device = RUNTIME_DEVICES.get(os.environ.get("SDC_RUNTIME", ""))
        if device is None:
            raise ValueError("Qwen 训练任务请在 nvidia 或 ascend 容器中运行。")
    return replace(config, training=replace(
        config.training, device=device,
        output_dir=config.training.output_dir.replace("{backend}", device),
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
    if args.world_size == 1 and args.n_gpu > 1:
        raise ValueError("实验按单卡运行：请只开放一张设备（多卡需用 torchrun 启动）。")
    return args


def resolve_checkpoint(output: Path, resume: str | None) -> str | None:
    """Pick the checkpoint to resume from, or verify the output dir is unused."""
    if resume is None:
        if output.exists() and any(output.iterdir()):
            raise ValueError("output_dir 非空：请换一个目录，或使用 --resume latest。")
        return None
    checkpoint = get_last_checkpoint(str(output)) if resume == "latest" and output.is_dir() else None
    if resume != "latest":
        checkpoint = str(resolve_path(resume))
    if not checkpoint or not (Path(checkpoint) / "trainer_state.json").is_file():
        raise ValueError("没有找到完整的 Trainer 检查点。")
    for name in CHECKPOINT_FILES:
        if not (Path(checkpoint) / name).is_file():
            raise ValueError(f"检查点缺少 {name}。")
    return checkpoint


def validate_resume(
    checkpoint: str, config: Config, data_hash: str, versions: dict[str, str], world_size: int,
) -> None:
    """Refuse to resume into different settings, data or software."""
    directory = Path(checkpoint)
    filename = directory.parent / "run.json"
    if not filename.is_file():
        raise ValueError("续训需要原始运行目录中的 run.json。")
    previous = json.loads(filename.read_text())
    current = {
        "config_signature": run_signature(config), "data_digest": data_hash,
        "packages": versions, "world_size": world_size,
        "lock_sha256": file_digest(ROOT / "uv.lock"),
    }
    for key, value in current.items():
        if previous.get(key) != value:
            raise ValueError(f"续训设置不一致：{key}。")
    state = json.loads((directory / "trainer_state.json").read_text())
    if state["global_step"] >= config.training.max_steps:
        raise ValueError("这个检查点已经达到 max_steps。")
    ranks = range(world_size) if world_size > 1 else [None]
    for rank in ranks:
        filename = directory / ("rng_state.pth" if rank is None else f"rng_state_{rank}.pth")
        if not filename.is_file():
            raise ValueError(f"检查点缺少 {filename.name}。")


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
    config: Config, args: TrainingArguments, datasets: Any, data_hash: str,
    versions: dict[str, str], checkpoint: str | None,
) -> dict[str, Any]:
    """Assemble the run.json provenance record."""
    device = getattr(torch, args.device.type, None)
    device_name = device.get_device_name(args.device) if hasattr(device, "get_device_name") else platform.machine()
    t = config.training
    return {
        "schema_version": 1, "task": "full_parameter_causal_language_modeling",
        "config_signature": run_signature(config), "data_digest": data_hash,
        "packages": versions, "world_size": args.world_size,
        "lock_sha256": file_digest(ROOT / "uv.lock"), "git_commit": _git_commit(),
        "python": platform.python_version(), "device": str(args.device), "device_name": device_name,
        "master_weights": "float32", "autocast": t.precision,
        "effective_batch_size": t.batch_size * t.gradient_accumulation_steps * args.world_size,
        "block_size": config.data.block_size,
        "train_sequences": len(datasets["train"]), "validation_sequences": len(datasets["validation"]),
        "training_budget": {
            "unit": "optimizer_updates", "max_steps": t.max_steps, "repeat_dataset": True,
        },
        "started_at": datetime.now(timezone.utc).isoformat(), "resume_from": checkpoint,
        "timing": "Trainer wall-clock metrics include the training loop; they are not GEMM kernel timings.",
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
        torch_dtype=torch.float32, attn_implementation=config.model.attention,
        trust_remote_code=False, local_files_only=offline,
    )
    model.config.use_cache = False
    return model


def train(
    config: Config,
    resume: str | None = None,
    offline: bool = False,
    callbacks: Sequence[TrainerCallback] = (),
) -> tuple[Trainer, dict[str, Any]]:
    """Run one training job and return the Trainer and its final metrics."""
    config = resolve_runtime_config(config)
    output = resolve_path(config.training.output_dir)
    checkpoint = resolve_checkpoint(output, resume)
    datasets, data_manifest = read_prepared(config)
    versions = package_versions()
    args = training_arguments(config)
    data_hash = prepared_digest(data_manifest)
    if checkpoint:
        validate_resume(checkpoint, config, data_hash, versions, args.world_size)
    # Every rank completes the output checks before rank 0 creates shared files.
    if args.world_size > 1:
        torch.distributed.barrier()
    session = uuid.uuid4().hex
    if args.process_index == 0:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "run.json", _run_record(
            config, args, datasets, data_hash, versions, checkpoint,
        ))
        write_json(output / "config.resolved.json", config.to_dict())
    if args.world_size > 1:
        torch.distributed.barrier()
    tokenizer = load_tokenizer(config, offline)
    trainer = Trainer(
        model=_load_model(config, offline), args=args, processing_class=tokenizer,
        train_dataset=datasets["train"], eval_dataset=datasets["validation"],
        data_collator=CausalCollator(), callbacks=[JsonMetrics(output, session), *callbacks],
    )
    try:
        result = trainer.train(resume_from_checkpoint=checkpoint)
        evaluation = trainer.evaluate(metric_key_prefix="validation")
        check_finite_loss({**result.metrics, **evaluation})
    except NonFiniteLossError as exc:
        if args.process_index == 0:
            write_json(output / "result.json", {
                "status": "failed", "reason": "nonfinite_loss", "message": str(exc),
                "global_step": trainer.state.global_step, "epoch": trainer.state.epoch,
                **exc.metrics,
            })
        raise
    loss = evaluation["validation_loss"]
    metrics = {
        **result.metrics, **evaluation, "status": "completed",
        "global_step": trainer.state.global_step, "epoch": trainer.state.epoch,
        "validation_perplexity": math.exp(loss) if math.isfinite(loss) and loss < 700 else None,
    }
    if args.process_index == 0:
        write_json(output / "result.json", metrics)
        trainer.save_state()
    return trainer, metrics
