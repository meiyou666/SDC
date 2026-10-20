import os
from pathlib import Path
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "environment/docker/nested/cdi.sh"


@pytest.mark.parametrize("failure", ["", "exit", "empty"])
def test_cdi_generation_is_atomic_and_keeps_unrelated_specs(tmp_path, failure):
    cdi = tmp_path / "cdi"
    cdi.mkdir()
    existing = cdi / "nvidia.yaml"
    existing.write_text("existing valid spec\n")
    unrelated = cdi / "other.yaml"
    unrelated.write_text("other device\n")
    (cdi / ".sdc-nvidia.old.yaml").write_text("broken remnant\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "nvidia-ctk"
    fake.write_text(
        '#!/usr/bin/env bash\nset -eu\n'
        '[[ "$1 $2 $3" == "cdi generate --output" ]]\n'
        'out=$4\n'
        # Reproduce the toolkit behavior that broke extensionless temp names.
        '[[ "$out" == *.yaml ]] || out="$out.yaml"\n'
        '[[ "$out" != "$SDC_NESTED_CDI_DIR/"* ]]\n'
        '[[ $(cat "$SDC_NESTED_CDI_DIR/nvidia.yaml") == "existing valid spec" ]]\n'
        'case "$FAILURE" in\n'
        'exit) echo partial > "$out"; exit 1;;\n'
        'empty) : > "$out";;\n'
        '*) echo "generated valid spec" > "$out";;\n'
        'esac\n'
    )
    fake.chmod(0o755)
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "SDC_NESTED_CONFIG": str(tmp_path / "missing.conf"),
            "SDC_NESTED_CDI_DIR": str(cdi),
            "SDC_NESTED_ETC": str(tmp_path / "etc"),
            "FAILURE": failure,
        },
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) == (not failure), result.stderr
    assert existing.read_text() == (
        "existing valid spec\n" if failure else "generated valid spec\n"
    )
    assert unrelated.read_text() == "other device\n"
    assert not list(cdi.glob(".sdc-nvidia.*"))
    assert not list(tmp_path.glob(".sdc-nvidia.*"))
    backups = list((tmp_path / "etc/cdi-backups").glob("*/.sdc-nvidia.old.yaml"))
    assert len(backups) == 1
    assert backups[0].read_text() == "broken remnant\n"


def test_child_exits_and_stops_other_children_when_buildkit_exits(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("dockerd", "buildkitd"):
        command = bin_dir / name
        command.write_text(
            '#!/usr/bin/env bash\n'
            f'echo $$ > "$PID_DIR/{name}.pid"\n'
            + ("sleep 0.2\nexit 0\n" if name == "buildkitd" else "exec sleep 60\n")
        )
        command.chmod(0o755)
    result = subprocess.run(
        ["bash", str(SCRIPT.with_name("runtime.sh")), "child"],
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "SDC_NESTED_CONFIG": str(tmp_path / "missing.conf"),
            "SDC_NESTED_BUILDKIT_BIN": str(bin_dir / "buildkitd"),
            "SDC_NESTED_LOG": str(tmp_path / "logs"),
            "PID_DIR": str(tmp_path),
        },
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode != 0
    assert (tmp_path / "logs/sdc-buildkit.log").is_file()
    for name in ("dockerd", "buildkitd"):
        pid = int((tmp_path / f"{name}.pid").read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
