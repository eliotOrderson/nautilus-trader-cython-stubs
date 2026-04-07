#!/usr/bin/env python3
"""
Parallel stub validation using multiprocessing.
Validates multiple .pyx/.pyi pairs concurrently.
"""

import argparse
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Import validation logic from existing module
from validate_stub import PyxPyiValidator


def count_any_types(pyi_file: Path) -> int:
    """Count occurrences of bare 'Any' type."""
    content = pyi_file.read_text(encoding="utf-8")
    # Match `: Any` but not `list[Any]` or `dict[str, Any]`
    pattern = r":\s*Any\s*($|=)"
    return len(re.findall(pattern, content, re.MULTILINE))


def validate_single_pair(args: tuple[Path, Path, bool]) -> tuple[str, bool, str, dict]:
    """Validate a single pyx/pyi pair. Returns (name, success, result_message, stats)."""
    pyx_file, pyi_file, pass_warning = args
    validator = PyxPyiValidator(pyx_file, pyi_file, pass_warning)

    # Run validation
    success = validator.validate()
    result_msg = validator.results()

    # Collect statistics
    stats = {
        "any_count": count_any_types(pyi_file),
        "missing_imports": len(validator.import_validator.validate()),
        "has_warnings": validator.reporter.has_warnings(),
    }

    return str(pyx_file), success, result_msg, stats


def main():
    parser = argparse.ArgumentParser(description="Validate Cython stubs in parallel.")
    parser.add_argument(
        "--workers",
        "-j",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)",
    )
    parser.add_argument(
        "--pass-warning", "-w", action="store_true", help="Pass even if warnings exist"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show validation output for each file",
    )
    parser.add_argument(
        "--check-quality",
        "-q",
        action="store_true",
        help="Check type quality and show quality report",
    )
    args = parser.parse_args()

    stubs_dir = Path("stubs")
    if not stubs_dir.exists():
        print("ERROR: stubs directory not found", file=sys.stderr)
        sys.exit(1)

    # Collect all pyx/pyi pairs
    validation_tasks = []
    for stub in sorted(stubs_dir.rglob("*.pyi")):
        rel_path = stub.relative_to(stubs_dir)
        target = (
            Path("nautilus_trader/nautilus_trader") / f"{rel_path.with_suffix('.pyx')}"
        )

        if not target.exists():
            print(f"ERROR: Missing implementation for stub: {stub}", file=sys.stderr)
            print(f"Expected: {target}", file=sys.stderr)
            continue

        validation_tasks.append((target, stub, args.pass_warning))

    if not validation_tasks:
        print("ERROR: No validation tasks found", file=sys.stderr)
        sys.exit(1)

    print(f"Validating {len(validation_tasks)} stub files...")

    # Run validations in parallel
    total = len(validation_tasks)
    passed = 0
    failed = []

    # Aggregate statistics
    total_stats = {
        "any_count": 0,
        "missing_imports": 0,
        "files_with_warnings": 0,
    }

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(validate_single_pair, task): task[0]
            for task in validation_tasks
        }

        for future in as_completed(futures):
            pyx_name, success, result_msg, stats = future.result()

            # Aggregate statistics
            total_stats["any_count"] += stats["any_count"]
            total_stats["missing_imports"] += stats["missing_imports"]
            if stats["has_warnings"]:
                total_stats["files_with_warnings"] += 1

            if success:
                passed += 1
                if args.verbose:
                    print(f"✅ {pyx_name}")
            else:
                failed.append((pyx_name, result_msg))
                if args.verbose:
                    print(f"❌ {pyx_name}")
                    print(result_msg)

    # Print summary
    print(f"\nTotal: {total}  Passed: {passed}  Failed: {total - passed}")

    if failed and not args.verbose:
        print("\nFailed validations:")
        for pyx_name, result_msg in failed:
            print(f"\n{pyx_name}:")
            print(result_msg)

    # Print quality report if requested
    if args.check_quality:
        print("\n=== Quality Report ===")
        print(f"Total 'Any' types: {total_stats['any_count']}")
        print(f"Missing imports: {total_stats['missing_imports']}")
        print(f"Files with warnings: {total_stats['files_with_warnings']}")

        # Quality score
        if total > 0:
            quality_score = (passed / total) * 100
            print(f"\nQuality Score: {quality_score:.1f}% ({passed}/{total} passed)")

            if total_stats["any_count"] > 0:
                avg_any = total_stats["any_count"] / total
                print(f"Average 'Any' types per file: {avg_any:.2f}")

            if total_stats["missing_imports"] > 0:
                print(
                    f"\n⚠️  Warning: {total_stats['missing_imports']} missing imports detected"
                )

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
