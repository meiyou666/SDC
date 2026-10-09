import os
from loguru import logger


def check_args_torchrun_main(args):
    if args.total_batch_size is None:
        args.gradient_accumulation = args.gradient_accumulation or 1
        args.total_batch_size = args.batch_size * args.gradient_accumulation

    assert args.total_batch_size % args.batch_size == 0, "total_batch_size must be divisible by batch_size"

    if args.max_train_tokens is not None:
        args.num_training_steps = args.max_train_tokens // args.total_batch_size
        logger.info(f"Training for {args.num_training_steps} update steps")

    if args.base_model_continue and args.base_model_path is not None:
        assert os.path.exists(args.base_model_path), f"--base_model_path={args.base_model_path} does not exist"

    if args.dtype in ["fp16", "float16"]:
        raise NotImplementedError("fp16 is not supported in torchrun_main.py. Use deepspeed_main.py instead (but it seems to have bugs)")

    if args.fi_nvbit_enable:
        valid_locations = {"forward", "backward", "forward_backward", "clipping", "optimizer"}
        if args.fi_nvbit_location not in valid_locations:
            raise ValueError(
                f"--fi_nvbit_location must be one of {sorted(valid_locations)} "
                f"when --fi_nvbit_enable is set, got: {args.fi_nvbit_location}"
            )
        if args.fi_nvbit_duration < 1:
            raise ValueError("--fi_nvbit_duration must be at least 1")

        has_explicit_steps = bool(
            args.fi_nvbit_steps
            and len(args.fi_nvbit_steps) > 0
            and -1 not in args.fi_nvbit_steps
        )
        if not has_explicit_steps and (
            args.fi_nvbit_trigger_rate is None or args.fi_nvbit_trigger_rate < 1
        ):
            raise ValueError(
                "Set --fi_nvbit_trigger_rate to a positive integer when "
                "--fi_nvbit_steps does not contain explicit update steps"
            )

    # Fail early with actionable errors: these assets cannot be downloaded in
    # offline mode, so continuing would only fail later with a less clear stack
    # trace from the dataloader or Transformers.
    for flag, path in (
        ("--train_data_path", args.train_data_path),
        ("--val_data_path", args.val_data_path),
    ):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{flag} must point to a local file, but it was not found: {path}")

    if not os.path.isdir(args.tokenizer_path):
        raise FileNotFoundError(
            f"--tokenizer_path must point to a local tokenizer directory, "
            f"but it was not found: {args.tokenizer_path}"
        )

    tokenizer_config = os.path.join(args.tokenizer_path, "tokenizer_config.json")
    if not os.path.isfile(tokenizer_config):
        raise FileNotFoundError(f"Tokenizer config not found: {tokenizer_config}")

    tokenizer_files = ("tokenizer.json", "spiece.model")
    if not any(os.path.isfile(os.path.join(args.tokenizer_path, name)) for name in tokenizer_files):
        raise FileNotFoundError(
            f"Tokenizer vocabulary not found in {args.tokenizer_path}. "
            f"Expected at least one of: {', '.join(tokenizer_files)}"
        )

    for recommended_file in ("spiece.model", "special_tokens_map.json"):
        recommended_path = os.path.join(args.tokenizer_path, recommended_file)
        if not os.path.isfile(recommended_path):
            logger.warning(f"Recommended tokenizer file not found: {recommended_path}")

    return args
