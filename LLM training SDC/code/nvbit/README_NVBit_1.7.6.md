# NVBit 1.7.6 adaptation notes

This folder is prepared for **NVBit 1.7.6**. The actual NVBit release (`1.7.6_nvbit_release`) must be present on the target machine because the core library (`libnvbit.a`) and headers are not redistributed in this repository.

## What changed from 1.7.4 to 1.7.6

| Version | Notable changes affecting this project |
|---------|----------------------------------------|
| 1.7.5   | CUDA 12.9 headers; fixes for `CALL.REL.NOINC`, patch-function argument passing, multithreaded kernel-launch serialization, `nvbit_tool_init()` per-context behavior, SASS string decoding. No API break for the existing tool. |
| 1.7.6   | CUDA 13.0 headers; added **SM_110** support; added `nvbit_dump_cubin()`; fixed `warpsync.collective`; fixed `nvbit_get_line_info()` issue (Hopper+ still needs manual cubin disassembly). |

The fault-injection tool (`tools/fault_injection/`) only uses stable APIs:

- `nvbit_get_related_functions`
- `nvbit_get_instrs`
- `nvbit_insert_call` / `nvbit_add_call_arg_*`
- `nvbit_enable_instrumented`
- `nvbit_read_reg` / `nvbit_write_reg`
- Signal handlers (`SIGUSR1` / `SIGUSR2`)

It does **not** use `nvbit_get_line_info()`, so the Hopper line-info limitation does not apply.

## Build

```bash
cd nvbit/1.7.6_nvbit_release/tools/fault_injection/
make clean && make
```

The default `ARCH?=sm_89` matches the original L40S experiments. If your cloud GPU has a different compute capability, override it:

```bash
make ARCH=sm_80
```

## Run

The shell scripts in `scripts/` source `scripts/configs/nvbit_default.sh`, which sets:

```bash
export LD_PRELOAD=./nvbit/1.7.6_nvbit_release/tools/fault_injection/fault_injection.so
```

If your NVBit 1.7.6 installation lives elsewhere, update that path.

## Requirements

Consult the official NVBit 1.7.6 README for the exact driver/CUDA matrix. The general requirements are:

- Linux
- CUDA >= 12.0
- `nvdisasm` in `PATH`
- SM compute capability supported by NVBit 1.7.6 (including sm_110)
