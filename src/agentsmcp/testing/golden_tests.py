#!/usr/bin/env python3
# -------------------------------------------------------------
# File: /Users/mikko/github/AgentsMCP/src/agentsmcp/testing/golden_tests.py
# -------------------------------------------------------------
"""
Golden Test Framework for the AgentsMCP role system.

Features
--------
* Load golden‑test JSON files from a directory tree.
* Automatic discovery of roles from the role registry.
* Partial / regexp matching for result envelopes.
* CLI integration for running tests manually.
* Extensible: add new roles or new comparison predicates with no code changes.
"""

import argparse
import json
import pathlib
import re
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ----------------------------------------------------------------
# Import the role system and envelope models
# ----------------------------------------------------------------
from agentsmcp.roles import ROLE_REGISTRY
from agentsmcp.models import (
    TaskEnvelopeV1,
    ResultEnvelopeV1,
    EnvelopeStatus,
)
from agentsmcp.roles.base import RoleName
from agentsmcp.roles.base import BaseRole  # For type hints in run_test()

# ----------------------------------------------------------------
# Helper types ---------------------------------------------------
# ----------------------------------------------------------------
@dataclass(frozen=True)
class ValidationResult:
    """Keeps the result of a single golden test."""
    passed: bool
    details: str = ""

@dataclass
class GoldenTestCase:
    """Encapsulates one golden‑test case."""
    name: str
    task: TaskEnvelopeV1
    expected: Dict[str, Any]
    # The location of the JSON file (useful for error reporting)
    file_path: Path

    @classmethod
    def load_from_file(cls, json_file: Path) -> "GoldenTestCase":
        """Load a single golden‑test JSON file."""
        with json_file.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        # Basic sanity checks
        if not isinstance(data, dict):
            raise ValueError(f"{json_file} does not contain a dict")

        if "task" not in data:
            raise KeyError(f"{json_file} missing 'task' section")
        if "expected" not in data:
            raise KeyError(f"{json_file} missing 'expected' section")

        # Convert the "task" dict to a TaskEnvelopeV1 (Pydantic v2)
        try:
            task = TaskEnvelopeV1.model_validate(data["task"])  # type: ignore[attr-defined]
        except Exception:
            # Fallback for older Pydantic versions
            task = TaskEnvelopeV1.parse_obj(data["task"])  # type: ignore[attr-defined]

        # The expected section is left as raw dict (will be interpreted by the matcher)
        expected = data["expected"]

        name = json_file.stem  # filename without suffix
        return cls(name=name, task=task, expected=expected, file_path=json_file)

# ----------------------------------------------------------------
# Matching utilities --------------------------------------------- 
# ----------------------------------------------------------------
def _match_value(pattern: Any, actual: Any) -> bool:
    """
    Recursively checks whether `actual` satisfies `pattern`.

    Supported patterns:
    * str: if it starts and ends with '/' it is treated as a regex.
    * List: must contain all elements of `pattern` (subset match).
    * Dict: all keys in pattern must exist in `actual` and match recursively.
    * Any other type: requires strict equality.
    """
    # Regex string pattern
    if isinstance(pattern, str) and pattern.startswith("/") and pattern.endswith("/"):
        return re.search(pattern[1:-1], actual or "") is not None

    # List/sub‑set matching
    if isinstance(pattern, list):
        if not isinstance(actual, list):
            return False
        # Every element of `pattern` should be present somewhere in `actual`
        return all(any(_match_value(p, a) for a in actual) for p in pattern)

    # Dictionary matching
    if isinstance(pattern, dict):
        if not isinstance(actual, dict):
            return False
        # All keys in pattern must exist in actual and satisfy recursively
        for key, sub_pattern in pattern.items():
            if key not in actual:
                return False
            if not _match_value(sub_pattern, actual[key]):
                return False
        return True

    # Final simple equality
    return pattern == actual


def validate_result(envelope: ResultEnvelopeV1, expected: Dict[str, Any]) -> ValidationResult:
    """
    Validate `envelope` against the expected pattern.

    Returns a ValidationResult containing a boolean flag and a human‑readable detail string.
    """
    mismatches: List[str] = []

    # Convert envelope to dict for simpler comparison
    actual_dict = envelope.model_dump()

    # A special case – we often want to allow the status to be one of several options
    if "status" in expected and isinstance(expected["status"], list):
        if envelope.status not in expected["status"]:
            mismatches.append(
                f"status={envelope.status!r} not in expected: {expected['status']}"
            )
        # remove after checking, because we already validated
        expected = {k: v for k, v in expected.items() if k != "status"}

    for key, pattern in expected.items():
        if key not in actual_dict:
            mismatches.append(f"Missing key in actual: {key}")
            continue
        if not _match_value(pattern, actual_dict[key]):
            mismatches.append(
                f"Key {key!r}: expected {pattern!r} but got {actual_dict[key]!r}"
            )

    passed = not mismatches
    details = "\n".join(mismatches) if mismatches else "All checks passed"
    return ValidationResult(passed=passed, details=details)


# ----------------------------------------------------------------
# Test runner ----------------------------------------------------
# ----------------------------------------------------------------
def run_test(case: GoldenTestCase, role_class: type) -> ValidationResult:
    """Instantiate the role, run the task, and compare the result."""
    instance: BaseRole = role_class()
    try:
        result: ResultEnvelopeV1 = instance.apply(case.task)
    except Exception as exc:
        tb = traceback.format_exc()
        return ValidationResult(
            passed=False,
            details=f"Role raised an exception: {exc!r}\nTraceback:\n{tb}",
        )

    return validate_result(result, case.expected)


def discover_golden_tests(
    root_dir: Path, role_filters: Optional[Iterable[str]] = None
) -> Dict[str, List[GoldenTestCase]]:
    """
    Walk the directory tree and collect all files ending in `.json`.

    Parameters
    ----------
    root_dir : Path
        The root directory containing the golden‑test folders.

    role_filters : Iterable[str], optional
        If supplied, only return cases belonging to these role names.

    Returns
    -------
    Dict[str, List[GoldenTestCase]]
        Mapping from role name to list of test cases.
    """
    golden_cases_by_role: Dict[str, List[GoldenTestCase]] = {}
    for json_file in root_dir.rglob("*.json"):
        case = GoldenTestCase.load_from_file(json_file)
        # The role name is stored by convention in the JSON under "expected -> role" (string)
        role_name: str = case.expected.get("role", None)
        if role_name is None:
            # Fallback: look into the path: top‑level folder = role name
            role_name = json_file.parent.name
        if role_filters and role_name not in role_filters:
            continue

        golden_cases_by_role.setdefault(role_name, []).append(case)

    return golden_cases_by_role


# ----------------------------------------------------------------
# CLI integration ------------------------------------------------
# ----------------------------------------------------------------
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Golden test runner for AgentsMCP roles"
    )
    parser.add_argument(
        "--json-dir",
        default="tests/golden",
        help="Root directory containing golden JSON test files",
    )
    parser.add_argument(
        "--role",
        action="append",
        help="Run tests only for this role (can be repeated)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print details for every test case, even if it passes",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    root_dir = Path(args.json_dir).resolve()
    if not root_dir.is_dir():
        print(f"❌ Directory not found: {root_dir}")
        sys.exit(1)

    golden_cases_by_role = discover_golden_tests(root_dir, args.role)

    if not golden_cases_by_role:
        print("❌ No golden test cases found")
        sys.exit(1)

    total_cases = 0
    total_passed = 0

    for role_name, case_list in golden_cases_by_role.items():
        role_class = ROLE_REGISTRY.get(RoleName(role_name))
        if role_class is None:
            print(f"⚠️  No role implementation registered for '{role_name}'. Skipping.")
            continue

        print(f"\n=== Running tests for role: {role_name} ({len(case_list)} cases) ===")
        for case in case_list:
            total_cases += 1
            result = run_test(case, role_class)
            status = "PASS" if result.passed else "FAIL"
            if result.passed:
                total_passed += 1
            print(f"{status:>5}: {case.file_path} ")
            if not result.passed or args.verbose:
                # Pretty‑print mismatches
                print("   ", result.details.replace("\n", "\n   "))
        print()

    print("=== Summary ===")
    print(f"Total tests   : {total_cases}")
    print(f"Passed        : {total_passed}")
    print(f"Failed        : {total_cases - total_passed}")
    if total_passed == total_cases:
        print("✅ All golden tests pass!")
        sys.exit(0)
    else:
        print("❌ Some golden tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
