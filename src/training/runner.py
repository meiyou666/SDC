"""Full-parameter training through the shared Transformers Trainer."""

from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import uuid

from accelerate.utils import is_npu_available
import torch
from transformers import AutoConfig, AutoModelForCausalLM, Trainer, TrainerCallback, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from .config import ROOT, file_digest, model_source, path, run_signature
from .data import CausalCollator, load_tokenizer, prepared_digest, read_prepared


def write_json(filename, value):
    filename = Path(filename)
    temporary = filename.with_suffix(filename.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(filename)


def package_versions():
    versions = {}
    for name in ("torch", "transformers", "datasets", "accelerate", "tokenizers", "torch-npu"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def require_device(device):
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this container; select the maintained CUDA environment.")
    if device == "npu" and not is_npu_available():
        raise ValueError("NPU is unavailable. The Ascend container needs matching CANN, torch_npu and device mounts.")


class JsonMetrics(TrainerCallback):
    def __init__(self, output, session):
        self.filename = output / "metrics.jsonl"
        self.session = session

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs:
            record = {"session": self.session, "step": state.global_step, **logs}
            with self.filename.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= state.max_steps:
            control.should_save = True
        return control


def training_arguments(config):
    t = config.training
    require_device(t.device)
    torch.set_num_threads(t.cpu_threads)
    args = TrainingArguments(
        output_dir=str(path(t.output_dir)),
        use_cpu=t.device == "cpu", bf16=t.precision == "bf16", fp16=t.precision == "fp16",
        tf32=False, seed=t.seed, data_seed=t.seed, full_determinism=t.full_determinism,
        max_steps=t.max_steps,
        per_device_train_batch_size=t.batch_size, per_device_eval_batch_size=t.eval_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        learning_rate=t.learning_rate, weight_decay=t.weight_decay, warmup_steps=t.warmup_steps,
        lr_scheduler_type="cosine", optim="adamw_torch", max_grad_norm=t.max_grad_norm,
        logging_strategy="steps", logging_steps=t.logging_steps, logging_first_step=True,
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
        raise ValueError(f"Requested {t.device}, but Trainer selected {args.device.type}.")
    if args.world_size == 1 and args.n_gpu > 1:
        raise ValueError("Launch multiple devices with torchrun, or expose one device for a single-process run.")
    return args


def resolve_checkpoint(output, resume):
    if resume is None:
        if output.exists() and any(output.iterdir()):
            raise ValueError("output_dir is not empty. Set a new directory or use --resume latest.")
        return None
    checkpoint = get_last_checkpoint(str(output)) if resume == "latest" and output.is_dir() else None
    if resume != "latest":
        checkpoint = str(path(resume))
    if not checkpoint or not (Path(checkpoint) / "trainer_state.json").is_file():
        raise ValueError("No complete Trainer checkpoint found.")
    for name in ("optimizer.pt", "scheduler.pt"):
        if not (Path(checkpoint) / name).is_file():
            raise ValueError(f"Checkpoint is missing {name}.")
    return checkpoint


def validate_resume(checkpoint, config, data_hash, versions, world_size):
    checkpoint = Path(checkpoint)
    filename = checkpoint.parent / "run.json"
    if not filename.is_file():
        raise ValueError("The checkpoint needs its original run.json.")
    previous = json.loads(filename.read_text())
    current = {
        "config_signature": run_signature(config), "data_digest": data_hash,
        "packages": versions, "world_size": world_size,
        "lock_sha256": file_digest(ROOT / "uv.lock"),
    }
    for key, value in current.items():
        if previous.get(key) != value:
            raise ValueError(f"Resume settings differ: {key}.")
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    if state["global_step"] >= config.training.max_steps:
        raise ValueError("This checkpoint has already reached max_steps.")
    ranks = range(world_size) if world_size > 1 else [None]
    for rank in ranks:
        filename = checkpoint / ("rng_state.pth" if rank is None else f"rng_state_{rank}.pth")
        if not filename.is_file():
            raise ValueError(f"Checkpoint is missing {filename.name}.")


def train(config, resume=None, offline=False, callbacks=()):
    output = path(config.training.output_dir)
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
        try:
            commit = subprocess.check_output(
                ["git", "-c", f"safe.directory={ROOT}", "rev-parse", "HEAD"],
                cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = None
        device = getattr(torch, args.device.type, None)
        device_name = device.get_device_name(args.device) if hasattr(device, "get_device_name") else platform.machine()
        record = {
            "schema_version": 1, "task": "full_parameter_causal_language_modeling",
            "config_signature": run_signature(config), "data_digest": data_hash,
            "packages": versions, "world_size": args.world_size,
            "lock_sha256": file_digest(ROOT / "uv.lock"), "git_commit": commit,
            "python": platform.python_version(), "device": str(args.device), "device_name": device_name,
            "master_weights": "float32", "autocast": config.training.precision,
            "effective_batch_size": config.training.batch_size * config.training.gradient_accumulation_steps * args.world_size,
            "block_size": config.data.block_size,
            "train_sequences": len(datasets["train"]), "validation_sequences": len(datasets["validation"]),
            "started_at": datetime.now(timezone.utc).isoformat(), "resume_from": checkpoint,
            "timing": "Trainer wall-clock metrics include the training loop; they are not GEMM kernel timings.",
        }
        write_json(output / "run.json", record)
        write_json(output / "config.resolved.json", config.to_dict())
    if args.world_size > 1:
        torch.distributed.barrier()
    tokenizer = load_tokenizer(config, offline)
    source = model_source(config.model)
    model_config = AutoConfig.from_pretrained(source, revision=config.model.revision, trust_remote_code=False, local_files_only=offline)
    limit = getattr(model_config, "max_position_embeddings", getattr(model_config, "n_positions", None))
    if limit is not None and config.data.block_size > limit:
        raise ValueError(f"block_size exceeds the model context length {limit}.")
    model = AutoModelForCausalLM.from_pretrained(
        source, revision=config.model.revision, config=model_config,
        torch_dtype=torch.float32, attn_implementation=config.model.attention,
        trust_remote_code=False, local_files_only=offline,
    )
    model.config.use_cache = False
    trainer = Trainer(
        model=model, args=args, processing_class=tokenizer,
        train_dataset=datasets["train"], eval_dataset=datasets["validation"],
        data_collator=CausalCollator(), callbacks=[JsonMetrics(output, session), *callbacks],
    )
    result = trainer.train(resume_from_checkpoint=checkpoint)
    evaluation = trainer.evaluate(metric_key_prefix="validation")
    loss = evaluation["validation_loss"]
    metrics = {
        **result.metrics, **evaluation, "global_step": trainer.state.global_step,
        "validation_perplexity": math.exp(loss) if math.isfinite(loss) and loss < 700 else None,
    }
    if args.process_index == 0:
        write_json(output / "result.json", metrics)
        trainer.save_state()
    return trainer, metrics
