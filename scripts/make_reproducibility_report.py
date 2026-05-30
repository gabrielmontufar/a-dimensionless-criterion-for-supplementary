from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "outputs" / "summaries" / "REPRODUCIBILITY_REPORT.json"


def write_report(payload: dict) -> None:
    check_date_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    if REPORT.exists():
        try:
            existing = json.loads(REPORT.read_text(encoding="utf-8-sig"))
            check_date_utc = existing.get("check_date_utc", check_date_utc)
        except json.JSONDecodeError:
            pass
    payload = {
        "package_version": "demand_polarity_map_v40_submission_reproducible",
        "check_date_utc": check_date_utc,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        **payload,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {REPORT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    write_report({"manual_invocation": True})
