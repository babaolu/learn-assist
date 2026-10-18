# BLINDING PROTOCOL

## Core Principle
The learner (repository owner) must not see item statements, solutions, keys, or rubrics before testing. Pre-exposure invalidates test integrity and distorts baseline and mastery evaluations.

## Rules for Autonomous Agents
1. **Zero Text Leakage**: Agents working on or with `data/itembank/` must NEVER print item statements, solution text, answer keys, or rubrics to the terminal, stdout, stderr, logs, or commit messages.
2. **Deterministic & Filtered Output**: Terminal outputs, tool outputs, and final reports must contain counts, item IDs, statuses, and file names only.
3. **Guard Enforcements**: Command-line interfaces and reporting scripts strictly guard against printing problem or key statements.

## Learner-Safe Interface
- **Safe to read**: Anything produced by `python -m learn_assist.itembank report <chapter>`. This command produces aggregated counts and statuses without revealing any underlying problem statements or keys.
- **Unsafe for the learner**: Directly inspecting `data/itembank/` files, viewing raw JSON files in items or keys directories, or reading diffs containing problem statements.

## Data Storage & Repository Boundaries
- **Item Bank Content**: All item statements, solutions, keys, and rubrics reside exclusively in `learn-assist-data/data/itembank/` (private repository).
- **Tooling and Validation**: The code and schema enforcement tools reside in `learn-assist` (`learn_assist/itembank.py`).
- **Tests**: All automated tests in `learn-assist/tests/` use synthetic fixtures only; real book content or exercise text must never appear in test files.
