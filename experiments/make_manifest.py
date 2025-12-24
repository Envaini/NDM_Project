import json
import platform
import subprocess
import sys
from pathlib import Path

def sh(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as e:
        return f"<error: {e}>"

def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "results" / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "project": "NDM_Project",
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "pip_freeze": sh(f'"{sys.executable}" -m pip freeze'),
        "git_head": sh("git rev-parse HEAD"),
        "git_status": sh("git status --porcelain"),
        "timestamp_local": sh("powershell -NoProfile -Command \"Get-Date -Format yyyy-MM-dd_HH-mm-ss\""),
    }

    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote:", out)

if __name__ == "__main__":
    main()
