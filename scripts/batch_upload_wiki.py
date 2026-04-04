#!/usr/bin/env python3
"""Batch upload Markdown documents from a directory tree to ToolRef.

Usage:
    python scripts/batch_upload_wiki.py --source-dir /tmp/claude-wiki --dry-run
    python scripts/batch_upload_wiki.py --source-dir /tmp/claude-wiki --api-url http://localhost:8000
    python scripts/batch_upload_wiki.py --source-dir /tmp/claude-wiki --retry-from failed.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx
from tqdm.asyncio import tqdm as async_tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_NAMESPACE = "claude"
DEFAULT_CONCURRENCY = 3
UPLOAD_ENDPOINT = "/api/v1/documents"

# Directories (by name) to skip during scanning.
SKIP_DIRS: frozenset[str] = frozenset(
    [
        "meta",
        "scripts",
        "17-Billing-Plans",
        "21-Account-Support",
        "16-Mobile-Desktop",
        "99-Other",
    ]
)

# Trailing hash suffix pattern: a dash followed by 7–12 lowercase hex characters
# at the very end of the stem, e.g. "Getting-Started-a1b2c3d4".
_HASH_SUFFIX_RE = re.compile(r"-[0-9a-f]{7,12}$", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ScanResult:
    """Outcome of scanning a source directory."""

    files: list[Path] = field(default_factory=list)
    skipped: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def parse_frontmatter(text: str) -> dict[str, str]:
    """Return a dict of YAML frontmatter key-value pairs (simple inline parser).

    Only handles flat string values (quoted or unquoted) on a single line.
    Does not depend on PyYAML so there are no extra install requirements.
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}

    result: dict[str, str] = {}
    for line in match.group(1).splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        key, _, raw_value = line.partition(":")
        key = key.strip()
        value = raw_value.strip().strip('"').strip("'")
        if key:
            result[key] = value
    return result


def stem_to_title(stem: str) -> str:
    """Convert a file stem to a human-readable title.

    Strips a trailing commit-hash-like suffix (e.g. ``-a1b2c3d``), then
    replaces hyphens and underscores with spaces and title-cases the result.

    Examples::

        >>> stem_to_title("01-Getting-Started-abc1234f")
        '01 Getting Started'
        >>> stem_to_title("Model_Context_Protocol")
        'Model Context Protocol'
    """
    clean = _HASH_SUFFIX_RE.sub("", stem)
    return clean.replace("-", " ").replace("_", " ").title()


def derive_title(path: Path, frontmatter: dict[str, str]) -> str:
    """Return the document title.

    Preference order:
    1. ``title`` field in YAML frontmatter.
    2. File stem with hash suffix stripped and humanised.
    """
    return frontmatter.get("title") or stem_to_title(path.stem)


def is_in_skip_dir(path: Path, source_root: Path) -> bool:
    """Return ``True`` if *path* resides inside one of :data:`SKIP_DIRS`."""
    try:
        relative = path.relative_to(source_root)
    except ValueError:
        return False
    # Inspect every ancestor component, excluding the filename itself.
    for part in relative.parts[:-1]:
        if part in SKIP_DIRS:
            return True
    return False


def collect_files(source_dir: Path) -> ScanResult:
    """Recursively collect all ``.md`` files under *source_dir*.

    Files whose ancestor directories match :data:`SKIP_DIRS` are excluded and
    counted separately in :attr:`ScanResult.skipped`.

    Returns:
        A :class:`ScanResult` with the accepted file list and the skip count.
    """
    accepted: list[Path] = []
    skipped = 0
    for p in sorted(source_dir.rglob("*.md")):
        if is_in_skip_dir(p, source_dir):
            skipped += 1
        else:
            accepted.append(p)
    return ScanResult(files=accepted, skipped=skipped)


# ---------------------------------------------------------------------------
# Upload logic
# ---------------------------------------------------------------------------


async def upload_file(
    client: httpx.AsyncClient,
    path: Path,
    namespace: str,
    api_url: str,
    semaphore: asyncio.Semaphore,
) -> tuple[Path, bool, str]:
    """Upload a single Markdown file to the ToolRef document API.

    The path is threaded through the return value so that callers can
    correlate results with the original files even when tasks complete out of
    order (``asyncio.as_completed`` wraps tasks in new Future objects that
    cannot be used as dict keys back to the originals).

    Args:
        client:     Shared :class:`httpx.AsyncClient` instance.
        path:       Absolute path to the Markdown file.
        namespace:  Target namespace in the ToolRef index.
        api_url:    Base URL of the ToolRef API (no trailing slash needed).
        semaphore:  Semaphore that caps the number of in-flight requests.

    Returns:
        A three-tuple ``(path, success, message)``.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    frontmatter = parse_frontmatter(text)
    title = derive_title(path, frontmatter)

    url = api_url.rstrip("/") + UPLOAD_ENDPOINT

    async with semaphore:
        try:
            response = await client.post(
                url,
                data={"namespace": namespace, "title": title},
                files={"file": (path.name, text.encode("utf-8"), "text/markdown")},
                timeout=30.0,
            )
            if response.status_code in (200, 201):
                return path, True, f"OK ({response.status_code})"
            return path, False, f"HTTP {response.status_code}: {response.text[:200]}"
        except httpx.RequestError as exc:
            return path, False, f"Request error: {exc}"


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


async def run(
    source_dir: Path,
    api_url: str,
    namespace: str,
    concurrency: int,
    dry_run: bool,
    retry_paths: Optional[list[Path]],
    failed_output: Path,
) -> None:
    """Orchestrate scanning, filtering, and concurrent uploading of documents.

    Args:
        source_dir:    Root directory to scan for Markdown files.
        api_url:       Base URL of the ToolRef API.
        namespace:     Target document namespace.
        concurrency:   Maximum number of simultaneous upload requests.
        dry_run:       When ``True``, only print what would be uploaded.
        retry_paths:   If provided, skip scanning and upload only these paths.
        failed_output: Destination file for failed-upload records (JSON).
    """
    skipped_count = 0

    if retry_paths is not None:
        files = retry_paths
        print(f"[retry] Re-uploading {len(files)} previously failed file(s).")
    else:
        result = collect_files(source_dir)
        files = result.files
        skipped_count = result.skipped
        print(
            f"[scan] Found {len(files)} Markdown file(s) under {source_dir} "
            f"({skipped_count} skipped)."
        )

    if not files:
        print("Nothing to upload.")
        _print_summary(total=0, success=0, failed=0, skipped=skipped_count)
        return

    if dry_run:
        print(
            f"\n[dry-run] Would upload {len(files)} file(s) "
            f"to namespace '{namespace}' at {api_url}"
        )
        preview = files[:10]
        for f in preview:
            text = f.read_text(encoding="utf-8", errors="replace")
            fm = parse_frontmatter(text)
            title = derive_title(f, fm)
            rel = f.relative_to(source_dir) if f.is_relative_to(source_dir) else f
            print(f"  {rel}  →  title='{title}'")
        if len(files) > 10:
            print(f"  … and {len(files) - 10} more")
        return

    semaphore = asyncio.Semaphore(concurrency)
    success_count = 0
    fail_count = 0
    failed_records: list[dict[str, str]] = []

    async with httpx.AsyncClient() as client:
        # Wrap each upload coroutine in a Task immediately so they are
        # scheduled and can start acquiring the semaphore right away.
        tasks: list[asyncio.Task[tuple[Path, bool, str]]] = [
            asyncio.ensure_future(
                upload_file(client, path, namespace, api_url, semaphore)
            )
            for path in files
        ]

        pbar = async_tqdm(total=len(tasks), desc="Uploading", unit="file")

        # ``asyncio.as_completed`` yields *wrapper* futures, not the original
        # Task objects — so we embed the path in the return value of
        # ``upload_file`` instead of relying on a task→path dict lookup.
        for future in asyncio.as_completed(tasks):
            original_path, ok, msg = await future
            pbar.update(1)
            if ok:
                success_count += 1
            else:
                fail_count += 1
                pbar.write(f"  FAIL {original_path}: {msg}")
                failed_records.append({"path": str(original_path), "reason": msg})

        pbar.close()

    # Persist or clean up failed.json.
    if failed_records:
        failed_output.write_text(
            json.dumps(failed_records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\n[failed] {len(failed_records)} failure(s) recorded to {failed_output}")
    elif failed_output.exists():
        # Remove a stale file from a previous run when everything succeeded.
        failed_output.unlink()

    _print_summary(
        total=len(files),
        success=success_count,
        failed=fail_count,
        skipped=skipped_count,
    )


def _print_summary(*, total: int, success: int, failed: int, skipped: int) -> None:
    """Print a formatted upload summary to stdout."""
    print(
        f"\n{'=' * 50}\n"
        f"Summary\n"
        f"  Total   : {total}\n"
        f"  Success : {success}\n"
        f"  Failed  : {failed}\n"
        f"  Skipped : {skipped}\n"
        f"{'=' * 50}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Batch upload Markdown files to the ToolRef document API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        type=Path,
        help="Root directory to scan for .md files.",
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help="Base URL of the ToolRef API.",
    )
    parser.add_argument(
        "--namespace",
        default=DEFAULT_NAMESPACE,
        help="Document namespace to upload into.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Maximum number of concurrent upload requests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and list files without uploading.",
    )
    parser.add_argument(
        "--retry-from",
        metavar="FAILED_JSON",
        type=Path,
        default=None,
        help="Path to a failed.json produced by a previous run; retry only those files.",
    )
    parser.add_argument(
        "--failed-output",
        type=Path,
        default=Path("failed.json"),
        help="Where to write failed upload records.",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    source_dir: Path = args.source_dir.expanduser().resolve()
    if not source_dir.is_dir():
        print(f"ERROR: --source-dir '{source_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    retry_paths: Optional[list[Path]] = None
    if args.retry_from is not None:
        retry_file: Path = args.retry_from
        if not retry_file.exists():
            print(f"ERROR: --retry-from '{retry_file}' does not exist.", file=sys.stderr)
            sys.exit(1)
        records: list[dict[str, str]] = json.loads(retry_file.read_text(encoding="utf-8"))
        retry_paths = [Path(r["path"]) for r in records]

    asyncio.run(
        run(
            source_dir=source_dir,
            api_url=args.api_url,
            namespace=args.namespace,
            concurrency=args.concurrency,
            dry_run=args.dry_run,
            retry_paths=retry_paths,
            failed_output=args.failed_output,
        )
    )
