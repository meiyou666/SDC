"""Prepare the shared Codex configuration without storing API keys in it."""

import json
import os
from pathlib import Path
import tomllib


def render_config(template, model=""):
    prefix = f"model = {json.dumps(model.strip(), ensure_ascii=False)}\n" if model.strip() else ""
    result = prefix + template
    tomllib.loads(result)
    return result


def main():
    template = Path(__file__).with_name("config.toml").read_text(encoding="utf-8")
    config = render_config(template, os.environ.get("CODEX_MODEL", ""))
    directory = Path(os.environ["CODEX_HOME"]) if os.environ.get("CODEX_HOME") else Path.home() / ".codex"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    destination = directory / "config.toml"
    destination.write_text(config, encoding="utf-8")
    destination.chmod(0o600)


if __name__ == "__main__":
    main()
