"""Run the configured small AEGIS reproduction: python -m algorithms.aegis --config YAML."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import subprocess

import torch
import yaml

from algorithms.aegis.aegis import Threshold, calibrate, prepare, sense
from algorithms.aegis.reference import DTYPES, Fault, bitwise_equal, gemm


def rate(k: int, n: int) -> dict:
    if n == 0:
        return dict(count=k, total=n, rate=None, ci95=None)
    z = 1.959963984540054
    p, d = k / n, 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return dict(count=k, total=n, rate=p, ci95=[center - radius, center + radius])


def run(config: dict) -> dict:
    expected = {'seed', 'shape', 'input_dtype', 'output_dtype', 'k_blocks', 'calibration_calls',
                'test_calls', 'alpha', 'threshold_floor', 'window', 'factor', 'columns',
                'targets', 'output_dir'}
    if config.keys() != expected:
        raise ValueError(f"configuration keys must be {sorted(expected)}")
    if (type(config['columns']) is not bool or not config['targets']
            or len(set(config['targets'])) != len(config['targets'])
            or set(config['targets']) - {'A', 'B', 'accumulator'}):
        raise ValueError("invalid columns or targets")
    for key in ('calibration_calls', 'test_calls'):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    floor = config['threshold_floor']
    if not math.isfinite(floor) or floor <= 0:
        raise ValueError("threshold_floor must be finite and positive")
    dtype, output_dtype = DTYPES[config['input_dtype']], DTYPES[config['output_dtype']]
    m, n, k = config['shape']
    generator = torch.Generator().manual_seed(config['seed'])

    def sample():
        a = torch.randn(m, k, generator=generator).to(dtype)
        b = torch.randn(k, n, generator=generator).to(dtype)
        return a, b

    def execute(a, b, fault=None):
        return gemm(a, b, output_dtype=output_dtype, k_blocks=config['k_blocks'], fault=fault)

    stats = {'acc32': [], 'output': []}
    for _ in range(config['calibration_calls']):
        a, b = sample()
        checksums = prepare(a, b, columns=config['columns'])
        acc, out = execute(a, b)
        for tap, value in (('acc32', acc), ('output', out)):
            stats[tap].append(sense(checksums, value).stat)
    policies = {}
    initial = {}
    for tap, adaptive in (('output', False), ('acc32', False), ('acc32', True)):
        name = f"{tap}-{'adaptive' if adaptive else 'fixed'}"
        policy = Threshold(max(floor, calibrate(stats[tap], config['alpha'])),
                           adaptive=adaptive, window=config['window'], factor=config['factor'])
        policies[name] = (tap, policy)
        initial[name] = policy.state_dict()
    events = []
    for record in range(config['test_calls']):
        a, b = sample()
        checksums = prepare(a, b, columns=config['columns'])
        _, clean = execute(a, b)
        faults = [None]
        for target in config['targets']:
            size = a.numel() if target == 'A' else b.numel() if target == 'B' else m * n
            coord = int(torch.randint(size, (), generator=generator).item())
            bits = 32 if target == 'accumulator' else a.element_size() * 8
            faults.append(Fault(target, 0, coord, zero_mask=True))
            faults.extend(Fault(target, bit, coord) for bit in range(bits))
        # Independent paired events: every method starts from the same calibrated
        # normal history for each event, not from the preceding injected fault.
        for fault in faults:
            acc, out = execute(a, b, fault)
            group = 'normal' if fault is None else 'zero_mask' if fault.zero_mask else 'fault'
            for name, (tap, policy) in policies.items():
                policy.load_state_dict(initial[name])
                evidence = sense(checksums, acc if tap == 'acc32' else out)
                threshold = policy.tau
                alarm = policy.update(evidence.stat)
                events.append(dict(record=record, method=name, group=group,
                                   fault=None if fault is None else asdict(fault),
                                   out_changed=not bitwise_equal(out, clean),
                                   output_finite=bool(torch.isfinite(out).all()), alarm=alarm,
                                   stat=evidence.stat if math.isfinite(evidence.stat) else None,
                                   stat_nonfinite=not math.isfinite(evidence.stat),
                                   threshold=threshold, state_after=policy.state_dict()))
    summary = {}
    for name in policies:
        selected = [e for e in events if e['method'] == name]
        normal = [e for e in selected if e['group'] == 'normal']
        zero = [e for e in selected if e['group'] == 'zero_mask']
        damaged = [e for e in selected if e['group'] == 'fault' and e['out_changed']]
        finite = [e for e in damaged if e['output_finite']]
        summary[name] = dict(false_alarm=rate(sum(e['alarm'] for e in normal), len(normal)),
                             zero_mask_alarm=rate(sum(e['alarm'] for e in zero), len(zero)),
                             detection=rate(sum(e['alarm'] for e in damaged), len(damaged)),
                             finite_detection=rate(sum(e['alarm'] for e in finite), len(finite)))
    return dict(config=config, initial_states=initial, summary=summary, events=events,
                execution='cpu-mathematical-reference', torch_version=torch.__version__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output = Path(config['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    destination = output / 'run.json'
    if destination.exists():
        raise FileExistsError(f"use a new output_dir: {destination}")
    result = run(config)
    root = Path(__file__).resolve().parents[3]
    result['uv_lock_sha256'] = hashlib.sha256((root / 'uv.lock').read_bytes()).hexdigest()
    result['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root,
                                                  text=True).strip()
    result['source_sha256'] = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).parent.glob('*.py'))
    }
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result['summary'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
