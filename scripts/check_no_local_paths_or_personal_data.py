from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

TEXT_SUFFIXES = {
    ".bib",
    ".cff",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".txt",
    ".yml",
    ".yaml",
}

FORBIDDEN_PATTERNS = {
    "windows_absolute_path": re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]"),
    "windows_user_path": re.compile(r"Users[\\/]"),
    "onedrive_path": re.compile(r"OneDrive", re.IGNORECASE),
    "local_username": re.compile(r"gjm31", re.IGNORECASE),
    "personal_email": re.compile(r"[\w.+-]+@(gmail|hotmail|outlook)\.com", re.IGNORECASE),
    "legacy_table_label": re.compile(r"Table V2"),
    "legacy_version_label": re.compile(r"\bv2[345]\b|experimental_database_validation_v2[345]|MRNB100 v18"),
    "pending_marker": re.compile(r"COMMIT_PENDING"),
}

ALLOWED_FILES = {
    "scripts/check_no_local_paths_or_personal_data.py",
    "scripts/check_docx_styles.py",
}


def rel_posix(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> None:
    failures: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = rel_posix(path)
        if rel in ALLOWED_FILES:
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="ignore")
        for name, pattern in FORBIDDEN_PATTERNS.items():
            if pattern.search(text):
                failures.append(f"{name}: {rel}")
    if failures:
        print("NO_LOCAL_PATHS_OR_LEGACY_LABELS: FAIL")
        for item in failures:
            print(item)
        raise SystemExit(1)
    print("NO_LOCAL_PATHS_OR_LEGACY_LABELS: PASS")


if __name__ == "__main__":
    main()
