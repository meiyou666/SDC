"""Load local API settings and use Codex's native API-key login."""

import os
from pathlib import Path
import subprocess
import sys


def main():
    environment = os.environ.copy()
    settings = Path("/workspace/SDC/.env")
    if settings.is_file():
        for line in settings.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            name, value = name.strip(), value.strip()
            if name not in {"OPENAI_API_KEY", "OPENAI_BASE_URL"}:
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            environment[name] = value

    key = environment.get("OPENAI_API_KEY", "").strip()
    if not key:
        print("Codex: fill OPENAI_API_KEY in .env, then run sh /opt/sdc/setup-codex.sh.")
        return 0
    result = subprocess.run(
        ["codex", "login", "--with-api-key"],
        input=key + "\n", env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode:
        print("Codex API login failed. Check the API settings in .env.", file=sys.stderr)
        return result.returncode
    print("Codex API login is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
