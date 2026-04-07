#!/usr/bin/env python3
"""
Parallel stub validation using multiprocessing.
Validates multiple .pyx/.pyi pairs concurrently.
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Import validation logic from existing module
from validate_stub import PyxPyiValidator


def validate_single_pair(args: tuple[Path, Path, bool]) -> tuple[str, bool, str]:
    """Validate a single pyx/pyi pair. Returns (name, success, result_message)."""
    pyx_file, pyi_file, pass_warning = args
    validator = PyxPyiValidator(pyx_file, pyi_file, pass_warning)
    success = validator.validate()
    result_msg = validator.results()
    return str(pyx_file), success, result_msg


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

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(validate_single_pair, task): task[0]
            for task in validation_tasks
        }

        for future in as_completed(futures):
            pyx_name, success, result_msg = future.result()

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

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
