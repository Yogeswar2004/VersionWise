import argparse
import re
import sys
from pathlib import Path

# Add server directory to sys.path if not present
SERVER_DIR = Path(__file__).resolve().parent.parent
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from app.config import settings
from app.ingestion import chunk_markdown, store_and_reconcile_chunks


def generate_canonical_url(base_url: str, rel_path: str) -> str:
    """Generate a canonical URL from base URL and relative file path.

    Normalizes index/README files to directory URLs while preserving distinct page slugs.
    """
    clean_base = base_url.rstrip("/") + "/"
    url_slug = rel_path[:-3] if rel_path.endswith(".md") else rel_path

    # Normalize standard documentation index filenames to directory slugs
    if url_slug.endswith("/index"):
        url_slug = url_slug[:-6]
    elif url_slug == "index":
        url_slug = ""
    elif url_slug.endswith("/README"):
        url_slug = url_slug[:-7]
    elif url_slug == "README":
        url_slug = ""

    return f"{clean_base}{url_slug}/" if url_slug else clean_base


def check_version_path_conflict(source_dir: Path, version: str) -> str | None:
    """Check if directory path components contain conflicting version indicators.

    Uses path-component and token-aware matching to avoid accidental substring matches.
    Returns an error message string if a conflict is detected, else None.
    """
    other_version = "2.12" if version == "1.10" else "1.10"
    conflicting_indicators = {other_version, f"v{other_version}"}

    for part in source_dir.resolve().parts:
        # Split part on common delimiters like '-', '_' or compare whole part
        tokens = set(re.split(r"[-_]", part)) | {part}
        if conflicting_indicators.intersection(tokens):
            return (
                f"Version conflict: Directory path '{source_dir}' contains component '{part}' "
                f"indicating version {other_version}, but --version {version} was requested."
            )
    return None


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VersionWise documentation ingestion CLI."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source-file", type=Path, help="Path to single markdown source file.")
    group.add_argument("--source-dir", type=Path, help="Path to directory containing markdown source files.")

    parser.add_argument("--library", type=str, default="pydantic", help="Target library name (default: pydantic).")
    parser.add_argument("--version", type=str, required=True, choices=["1.10", "2.12"], help="Product target version.")
    parser.add_argument("--url", type=str, required=True, help="Canonical source URL (or base URL for directory).")
    parser.add_argument("--source-release", type=str, default=None, help="Exact upstream release tag.")
    parser.add_argument("--source-commit", type=str, default=None, help="Exact upstream Git commit SHA.")
    parser.add_argument("--dry-run", action="store_true", help="Perform in-memory text parsing and validation without loading models or connecting to MongoDB.")
    parser.add_argument("--dry-run-embeddings", action="store_true", help="Perform in-memory parsing, validation, and embedding generation without connecting to MongoDB.")
    parser.add_argument("--embedding-provider", choices=["sentence-transformers", "mock"], default=None, help="Override configured embedding provider.")
    parser.add_argument("--allow-empty-source", action="store_true", help="Allow processing empty/whitespace-only files without error.")

    return parser.parse_args(args)


def process_file(
    file_path: Path,
    canonical_url: str,
    library: str,
    version: str,
    source_release: str | None,
    source_commit: str | None,
    allow_empty_source: bool,
) -> tuple[str, list]:
    text = file_path.read_text(encoding="utf-8")
    page_title = file_path.stem.replace("_", " ").title()

    chunks = chunk_markdown(
        text=text,
        canonical_url=canonical_url,
        page_title=page_title,
        version=version,
        library=library,
        source_release=source_release,
        source_commit=source_commit,
        allow_empty_source=allow_empty_source,
    )
    return page_title, chunks


def main(args: list[str] | None = None) -> int:
    parsed = parse_args(args)

    # 1. Discover target files & build complete URL mapping
    targets: list[tuple[Path, str]] = []
    if parsed.source_file:
        if not parsed.source_file.is_file():
            print(f"Error: File not found: {parsed.source_file}", file=sys.stderr)
            return 1
        targets.append((parsed.source_file, parsed.url))
    else:
        if not parsed.source_dir.is_dir():
            print(f"Error: Directory not found: {parsed.source_dir}", file=sys.stderr)
            return 1

        # Check for directory path version conflicts
        conflict_err = check_version_path_conflict(parsed.source_dir, parsed.version)
        if conflict_err:
            print(f"Error: {conflict_err}", file=sys.stderr)
            return 1

        base_url = parsed.url.rstrip("/") + "/"
        seen_urls: dict[str, Path] = {}
        for md_file in sorted(parsed.source_dir.rglob("*.md")):
            rel_path = md_file.relative_to(parsed.source_dir).as_posix()
            doc_url = generate_canonical_url(base_url, rel_path)
            if doc_url in seen_urls:
                print(
                    f"Error: Duplicate canonical URL '{doc_url}' generated by both "
                    f"'{seen_urls[doc_url]}' and '{md_file}'. Aborting.",
                    file=sys.stderr,
                )
                return 1
            seen_urls[doc_url] = md_file
            targets.append((md_file, doc_url))

    if not targets:
        print("No markdown files found to process.")
        return 0

    # 2. Local in-memory parsing and validation
    all_chunks_by_file: list[tuple[Path, str, str, list]] = []
    for file_path, doc_url in targets:
        try:
            page_title, chunks = process_file(
                file_path=file_path,
                canonical_url=doc_url,
                library=parsed.library,
                version=parsed.version,
                source_release=parsed.source_release,
                source_commit=parsed.source_commit,
                allow_empty_source=parsed.allow_empty_source,
            )
            all_chunks_by_file.append((file_path, doc_url, page_title, chunks))
        except Exception as e:
            print(f"Validation Error processing {file_path}: {e}", file=sys.stderr)
            return 1

    # 3. Dry-Run Isolation Check (Text-Only)
    if parsed.dry_run:
        print("=== VersionWise Documentation Ingestion [TEXT DRY RUN] ===")
        print(f"Library: {parsed.library} | Version: {parsed.version} | Files: {len(all_chunks_by_file)}")
        print("Notice: In-memory parsing only. Zero models loaded, zero embeddings generated, zero database connections.\n")

        for file_path, doc_url, page_title, chunks in all_chunks_by_file:
            print(f"--- File: {file_path.name} ---")
            print(f"  URL: {doc_url}")
            print(f"  Title: {page_title}")
            print(f"  Chunks generated: {len(chunks)}")
            for c in chunks[:3]:
                print(f"    [{c.chunk_index}] ID: {c.chunk_id[:12]}... | Hash: {c.content_hash[:8]}... | Breadcrumb: {c.heading_path} ({c.char_count} chars)")
            if len(chunks) > 3:
                print(f"    ... and {len(chunks) - 3} more chunks.")
            print()

        print("[DRY RUN COMPLETE] All inputs successfully chunked and validated.")
        return 0

    # 4. Dry-Run with Embeddings Mode
    from app.embeddings import get_embedding_provider

    active_provider_name = (parsed.embedding_provider or settings.embedding_provider).strip().lower()

    if parsed.dry_run_embeddings:
        print("=== VersionWise Documentation Ingestion [EMBEDDINGS DRY RUN] ===")
        print(f"Library: {parsed.library} | Version: {parsed.version} | Files: {len(all_chunks_by_file)}")
        print(f"Embedding Provider: {active_provider_name} | Notice: Zero database connections, writes, or deletions.\n")

        try:
            dry_run_settings = settings.model_copy(update={"embedding_provider": active_provider_name})
            provider = get_embedding_provider(dry_run_settings)

            for file_path, doc_url, page_title, chunks in all_chunks_by_file:
                for c in chunks:
                    c.embedding = provider.embed_chunk(c)

                print(f"--- File: {file_path.name} ---")
                print(f"  URL: {doc_url}")
                print(f"  Title: {page_title}")
                print(f"  Chunks embedded: {len(chunks)}")
                for c in chunks[:3]:
                    vec_len = len(c.embedding) if c.embedding else 0
                    print(f"    [{c.chunk_index}] ID: {c.chunk_id[:12]}... | Hash: {c.content_hash[:8]}... | Embedding: {vec_len} dims | Breadcrumb: {c.heading_path}")
                if len(chunks) > 3:
                    print(f"    ... and {len(chunks) - 3} more chunks.")
                print()

            print("[EMBEDDINGS DRY RUN COMPLETE] All inputs successfully chunked, embedded, and validated in memory.")
            return 0
        except Exception as exc:
            print(f"Dry-Run Embedding Error: {exc}", file=sys.stderr)
            return 1

    # 5. Live Persistence Execution (only when not in dry-run mode)
    # Strict mock-provider guard: refuse to persist mock embeddings to MongoDB
    if active_provider_name == "mock":
        print(
            "Fatal Error: Mock embeddings cannot be persisted to any database. "
            "Mock embeddings are strictly restricted to unit tests and --dry-run-embeddings.",
            file=sys.stderr,
        )
        return 1

    # Compute and validate embeddings for complete active batch in memory before database operations
    try:
        live_settings = settings.model_copy(update={"embedding_provider": active_provider_name})
        provider = get_embedding_provider(live_settings)
        for file_path, doc_url, page_title, chunks in all_chunks_by_file:
            for c in chunks:
                c.embedding = provider.embed_chunk(c)
    except Exception as exc:
        print(f"Embedding Generation Error before persistence: {exc}", file=sys.stderr)
        return 1

    from app.database import create_mongo_client

    client = create_mongo_client(settings)
    if client is None:
        print("Error: MongoDB URI is not configured in settings. Ingestion halted.", file=sys.stderr)
        return 1

    try:
        db = client[settings.mongodb_database]
        any_failure = False

        print(f"=== Beginning Ingestion into database: {settings.mongodb_database} ===")
        for file_path, doc_url, page_title, chunks in all_chunks_by_file:
            summary = store_and_reconcile_chunks(
                db=db,
                chunks=chunks,
                library=parsed.library,
                version=parsed.version,
                canonical_url=doc_url,
                allow_empty_source=parsed.allow_empty_source,
            )
            status_text = "SUCCESS" if summary.success else "FAILED"
            print(
                f"[{status_text}] {file_path.name} -> "
                f"Inserted: {summary.chunks_inserted}, "
                f"Updated: {summary.chunks_updated}, "
                f"Skipped: {summary.chunks_skipped}, "
                f"Deleted: {summary.chunks_deleted}"
            )
            if not summary.success:
                any_failure = True
                print(f"  Error: {summary.error_message}")

        return 1 if any_failure else 0
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
