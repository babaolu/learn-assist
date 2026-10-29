"""
Tests verifying and enforcing that the learn-assist test runner and forms tools
use NO AI models, NO network access, and NO API keys or tokens.
"""

import ast
import json
import os
import socket
from pathlib import Path
from typing import List, Set, Tuple

import pytest

from learn_assist.forms import make_forms
from learn_assist.testrun import (
    RUNNER_VERSION,
    format_statement,
    get_testrun_status,
    run_sitting,
)
from tests.test_forms_testrun import create_synthetic_frozen_bank


BANNED_MODULES = [
    "requests",
    "urllib",
    "http",
    "socket",
    "subprocess",
    "google.generativeai",
    "google.genai",
    "google",
    "anthropic",
    "openai",
]


def get_transitive_in_repo_modules(entry_files: List[Path], repo_root: Path) -> Set[Path]:
    """
    Computes the transitive closure of in-repo Python files imported by entry_files.
    """
    visited: Set[Path] = set()
    queue: List[Path] = [p.resolve() for p in entry_files]

    while queue:
        current_file = queue.pop()
        if current_file in visited:
            continue
        visited.add(current_file)

        if not current_file.exists():
            continue

        tree = ast.parse(current_file.read_text(encoding="utf-8"), filename=str(current_file))
        for node in ast.walk(tree):
            mod_name = None
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("learn_assist"):
                        mod_name = alias.name
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("learn_assist"):
                    mod_name = node.module

            if mod_name:
                rel_parts = mod_name.split(".")
                candidate_file = (repo_root / "/".join(rel_parts)).with_suffix(".py")
                if candidate_file.exists():
                    resolved = candidate_file.resolve()
                    if resolved not in visited:
                        queue.append(resolved)

    return visited


def scan_file_for_banned_constructs(file_path: Path) -> List[str]:
    """
    Scans a Python file's AST for banned imports, system calls, and KEY/TOKEN env reads.
    Returns a list of violation descriptions.
    """
    violations: List[str] = []
    tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))

    for node in ast.walk(tree):
        # 1. Banned imports (import foo)
        if isinstance(node, ast.Import):
            for alias in node.names:
                for banned in BANNED_MODULES:
                    if alias.name == banned or alias.name.startswith(f"{banned}."):
                        violations.append(
                            f"L{node.lineno}: Banned import '{alias.name}' in {file_path.name}"
                        )

        # 2. Banned imports from (from foo import bar)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                for alias in node.names:
                    full_name = f"{node.module}.{alias.name}"
                    for banned in BANNED_MODULES:
                        if (
                            node.module == banned
                            or node.module.startswith(f"{banned}.")
                            or full_name == banned
                            or full_name.startswith(f"{banned}.")
                        ):
                            violations.append(
                                f"L{node.lineno}: Banned import '{full_name}' in {file_path.name}"
                            )

        # 3. Banned function / method calls
        elif isinstance(node, ast.Call):
            # os.system(...)
            if isinstance(node.func, ast.Attribute):
                val = node.func.value
                if isinstance(val, ast.Name) and val.id == "os" and node.func.attr == "system":
                    violations.append(f"L{node.lineno}: Call to os.system in {file_path.name}")
                elif isinstance(val, ast.Name) and val.id in ("subprocess", "socket"):
                    violations.append(
                        f"L{node.lineno}: Call to {val.id}.{node.func.attr} in {file_path.name}"
                    )

                # os.environ.get('...KEY...') or os.getenv('...KEY...')
                if node.func.attr in ("get", "getenv"):
                    if node.args and isinstance(node.args[0], ast.Constant):
                        arg_val = str(node.args[0].value)
                        if "KEY" in arg_val.upper() or "TOKEN" in arg_val.upper():
                            violations.append(
                                f"L{node.lineno}: Read of sensitive env var '{arg_val}' in {file_path.name}"
                            )

        # 4. os.environ['...KEY...']
        elif isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Attribute) and node.value.attr == "environ":
                if isinstance(node.slice, ast.Constant):
                    slice_val = str(node.slice.value)
                    if "KEY" in slice_val.upper() or "TOKEN" in slice_val.upper():
                        violations.append(
                            f"L{node.lineno}: Read of sensitive env var '{slice_val}' in {file_path.name}"
                        )

    return violations


def test_static_ast_no_models_or_network():
    """
    Statically scans testrun.py, forms.py, and all transitive in-repo imports.
    Fails if any banned module, system call, or KEY/TOKEN env read is present.
    """
    repo_root = Path(__file__).resolve().parent.parent
    entry_files = [
        repo_root / "learn_assist" / "testrun.py",
        repo_root / "learn_assist" / "forms.py",
    ]

    transitive_files = get_transitive_in_repo_modules(entry_files, repo_root)
    assert len(transitive_files) >= 2, "Failed to locate entry files in transitive scan"

    all_violations: List[str] = []
    for f in sorted(transitive_files):
        violations = scan_file_for_banned_constructs(f)
        all_violations.extend(violations)

    assert not all_violations, "Banned constructs found in runner code path:\n" + "\n".join(
        all_violations
    )


def test_static_ast_scanner_catches_violations():
    """
    Verifies that the static scanner correctly flags banned imports, calls, and env vars.
    """
    sample_bad_code = """
import requests
import subprocess
from urllib import request
import socket
import openai
import anthropic
from google import genai
import os

os.system("echo hacked")
subprocess.run(["ls"])
k1 = os.environ.get("OPENAI_API_KEY")
k2 = os.getenv("GITHUB_TOKEN")
k3 = os.environ["MY_SECRET_KEY"]
"""
    tmp_file = Path("test_bad_sample.py")
    try:
        tmp_file.write_text(sample_bad_code, encoding="utf-8")
        violations = scan_file_for_banned_constructs(tmp_file)
        assert any("requests" in v for v in violations)
        assert any("subprocess" in v for v in violations)
        assert any("urllib" in v for v in violations)
        assert any("socket" in v for v in violations)
        assert any("openai" in v for v in violations)
        assert any("anthropic" in v for v in violations)
        assert any("genai" in v for v in violations)
        assert any("os.system" in v for v in violations)
        assert any("OPENAI_API_KEY" in v for v in violations)
        assert any("GITHUB_TOKEN" in v for v in violations)
        assert any("MY_SECRET_KEY" in v for v in violations)
    finally:
        if tmp_file.exists():
            tmp_file.unlink()


def test_runtime_offline_sitting_with_socket_blocked(tmp_path, monkeypatch):
    """
    Runs a full synthetic test sitting with socket.socket and network functions
    monkeypatched to raise an exception. Proves no network access happens at runtime.
    """
    # Block socket entirely
    def fail_socket(*args, **kwargs):
        raise RuntimeError("Network access attempted via socket!")

    monkeypatch.setattr(socket, "socket", fail_socket)
    monkeypatch.setattr(socket, "create_connection", fail_socket)
    monkeypatch.setattr(socket, "getaddrinfo", fail_socket)
    monkeypatch.setattr(socket, "gethostbyname", fail_socket)

    # 1. Setup synthetic frozen bank
    create_synthetic_frozen_bank(tmp_path, chapter=1)

    # 2. Generate forms
    forms_res = make_forms(chapter=1, seed=42, data_dir=tmp_path)
    assert "baseline" in forms_res["forms"]
    sittings = forms_res["forms"]["baseline"]["sittings"]
    assert len(sittings) >= 1

    # 3. Administer sitting 1 with synthetic input function (no stdin)
    input_answers = iter(["synthetic answer 1", "synthetic answer 2", "dont know", "42", "skip", "final reasoning"])
    rc = run_sitting(
        chapter=1,
        form="baseline",
        sitting_num=1,
        data_dir=tmp_path,
        interactive=False,
        input_func=lambda: next(input_answers, "default answer"),
    )
    assert rc == 0, "run_sitting failed under offline conditions"

    # 4. Verify local response files
    resp_dir = tmp_path / "responses" / "ch01" / "baseline"
    assert resp_dir.exists()
    sitting_item_ids = sittings[0]["item_ids"]
    assert len(sitting_item_ids) > 0

    for item_id in sitting_item_ids:
        resp_file = resp_dir / f"{item_id}.json"
        assert resp_file.exists(), f"Response file {resp_file} was not written locally"
        data = json.loads(resp_file.read_text(encoding="utf-8"))
        assert data["item_id"] == item_id
        assert data["form"] == "baseline"
        assert data["sitting"] == 1
        assert "raw_answer" in data
        assert "seconds" in data
        assert "shown_at" in data
        assert "submitted_at" in data
        assert data["runner_version"] == RUNNER_VERSION

    # 5. Check testrun status report
    status = get_testrun_status(1, data_dir=tmp_path)
    assert "error" not in status
    base_status = status["forms"]["baseline"]
    assert base_status["answered"] + base_status["skipped"] == len(sitting_item_ids)


def test_no_key_or_token_env_vars_read_at_runtime(tmp_path, monkeypatch):
    """
    Instruments environment variable access and verifies that NO variable containing
    'KEY' or 'TOKEN' is read during form generation or sitting execution.
    """
    accessed_env_vars: Set[str] = set()

    # Set tempting dummy keys in environment first
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-anthropic-key")
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-gemini-key")
    monkeypatch.setenv("HF_TOKEN", "dummy-hf-token")
    monkeypatch.setenv("API_KEY", "dummy-key")

    tracking_active = False

    def spy_getitem(self, key):
        if tracking_active:
            accessed_env_vars.add(key)
            if "KEY" in key.upper() or "TOKEN" in key.upper():
                raise AssertionError(f"Forbidden environment variable accessed: {key}")
        return orig_getitem(self, key)

    def spy_get(self, key, default=None):
        if tracking_active:
            accessed_env_vars.add(key)
            if "KEY" in key.upper() or "TOKEN" in key.upper():
                raise AssertionError(f"Forbidden environment variable accessed: {key}")
        return orig_get(self, key, default)

    def spy_getenv(key, default=None):
        if tracking_active:
            accessed_env_vars.add(key)
            if "KEY" in key.upper() or "TOKEN" in key.upper():
                raise AssertionError(f"Forbidden environment variable accessed: {key}")
        return orig_getenv(key, default)

    monkeypatch.setattr(os.environ.__class__, "__getitem__", spy_getitem)
    monkeypatch.setattr(os.environ.__class__, "get", spy_get)
    monkeypatch.setattr(os, "getenv", spy_getenv)

    # Run complete workflow with tracking active
    create_synthetic_frozen_bank(tmp_path, chapter=1)
    tracking_active = True
    try:
        make_forms(chapter=1, seed=42, data_dir=tmp_path)
        run_sitting(
            chapter=1,
            form="baseline",
            sitting_num=1,
            data_dir=tmp_path,
            interactive=False,
            input_func=lambda: "answer",
        )
        get_testrun_status(1, data_dir=tmp_path)
    finally:
        tracking_active = False

    # Assert no sensitive env var was ever read
    sensitive_reads = [k for k in accessed_env_vars if "KEY" in k.upper() or "TOKEN" in k.upper()]
    assert not sensitive_reads, f"Sensitive env vars were read: {sensitive_reads}"

