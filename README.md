# learn-assist

Adaptive tutoring system — teaches a textbook chapter to mastery and verifies retention with a delayed cold test.

## Overview

The core purpose of `learn-assist` is to test one central empirical question: can an adaptive tutoring system teach a textbook chapter well enough that the learner understands it without ever reading the source? Everything else is deferred until that is answered.

The target experiment focuses on Chapter 3 ("Signal Generation and Detection") of Liang & Lauterbur, *Principles of Magnetic Resonance Imaging* (IEEE Press, 2000).

## Lineage & Reusable Components

`learn-assist` is derived from `book2Lecture`. It strips the lecture generation pipeline while carrying forward the local OCR microservice (`services/ocr_service/` and `services/ocr_client.py`). Because textbook mathematical notation in scanned PDFs contains corrupted OCR text layers (mangled Greek letters and operators), mathematical pages are transcribed and verified using vision models and local OCR before downstream stages run.

## Roadmap & Stages

The project is structured in 4 sequential stages:

- **Stage 1: Ingestion and verification** (Current Stage) — High-fidelity text and mathematical formula extraction from textbook scans, verified before anything downstream runs.
- **Stage 2: Item-bank generation** — Creating Bucket A (end-of-chapter book problems) and Bucket B (transfer questions satisfying strict multi-concept, inversion, or domain-shift criteria) with dual-model verification and difficulty screening.
- **Stage 3: Tutoring loop** — Interactive teaching sequence using pre-computed Socratic question banks, rubrics, and lightweight on-device (3B class) model execution with escalation on uncertainty.
- **Stage 4: Scoring** — Cold baseline comparison, delayed post-testing (minimum 7-day gap), dual-model independent grading, and separate reporting of Bucket A and Bucket B scores.

## Project Structure

```
learn-assist/
├── book/            # Source PDFs (gitignored, never committed)
├── data/            # Extracted text, manifests, item banks, logs (gitignored)
├── docs/            # Project documentation and specifications (see docs/DESIGN.md)
├── learn_assist/    # Application code
├── services/        # Local OCR microservice and client carried forward from book2Lecture
└── tests/           # Test suites
```

For full system specifications and governing rules, see [docs/DESIGN.md](docs/DESIGN.md).
