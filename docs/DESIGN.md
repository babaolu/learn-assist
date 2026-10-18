# PURPOSE
Test one question: can a system teach a chapter well enough that the learner
understands it without ever reading the source? Everything else is deferred
until that is answered.

# THE EXPERIMENT
Target: Chapter 3, "Signal Generation and Detection", of Liang & Lauterbur,
Principles of Magnetic Resonance Imaging (IEEE Press, 2000).
1. Prerequisite diagnostic on Chapter 2 (Mathematical Fundamentals). If the
   learner is not solid, Chapter 2 becomes the target instead.
2. Cold baseline test on a chapter not studied, including transfer-style
   items, to establish what the learner scores from general knowledge alone.
3. Interactive teaching of Chapter 3. The learner never opens the book.
4. Minimum 7-day gap, during which other material may be studied.
5. Final test. Compare to baseline.

# MASTERY RULE
A concept is mastered when the learner explains it correctly unaided AND
applies it to a previously unseen problem, on two different days. No
probabilistic mastery score. Bayesian Knowledge Tracing is explicitly
rejected: its parameters cannot be fitted for a single learner, and published
held-out AUC on real datasets runs 0.51-0.57.

# TEACHING SOURCE
Union of two sources: the book's own section structure, and a model-extracted
concept manifest. Over-teaching is acceptable; under-teaching is not.

# TEST COMPOSITION — two buckets, scored and reported SEPARATELY, never blended
Bucket A, Book problems: the author's own end-of-chapter problems. Floor check.
Bucket B, Transfer questions: generated. Each must satisfy at least TWO of:
  1. combines two or more concepts the book treats in separate sections
  2. inverts the given and the unknown relative to any book problem
  3. applies the concept outside the domain the book works in
  4. requires noticing missing, redundant or inconsistent information
  5. requires explaining why a standard method fails in an edge case
  6. requires a quantitative result the book never states
Every Bucket B item stores an anchor: the specific book problem it exceeds,
and which criteria apply.
Difficulty screen: the strongest available model solves both the anchor
problem and the transfer question. If the transfer solution is shorter or
more confident than the anchor solution, flag the item instead of scoring it.
Ratio: each major concept gets at least one Bucket A item where one exists
and at least one Bucket B item. Target 60/40 A:B by count, minimum 5 B items.
The headline result is the Bucket B score.

# TEST INTEGRITY
- Free response only. No multiple choice.
- Two models from DIFFERENT families independently write questions and answer
  each other's from the source text. Agreement means the item is sound;
  disagreement surfaces to the user; an unanswerable item is discarded.
- Grading uses a rubric derived from the source text. Both models grade
  independently; disagreements surface to the user rather than being averaged.
- Raw learner answers are stored verbatim for later re-grading.
- Test items are never shown during teaching.
- Blinding protocol: see [docs/BLINDING.md](BLINDING.md) for strict blinding rules ensuring the learner never sees items, solutions, keys, or rubrics prior to testing.

# INGESTION
The source PDF is a scan. Its OCR text layer is corrupted on mathematics —
Greek letters and operators are mangled. The text layer alone is NOT usable.
Mathematical pages must be transcribed by a vision model reading page images.
Extraction must be verified before anything downstream runs.

# ARCHITECTURE
Per chapter, once, on a large model, cached to disk: concept manifest,
explanations, Socratic question sequence with expected answers, grading
rubrics, worked examples.
Per session, on-device (3B class): delivering questions, first-pass matching
of answers against pre-written rubrics, session state. Escalates anything
uncertain rather than scoring it.
No model at all: prerequisite graph, ordering, scheduling, logs — SQLite and
ordinary code.
The model endpoint is configuration, not a hard-wired choice.

# NON-NEGOTIABLE ENGINEERING RULE
A requirement stated only in a prompt is not implemented. Every rule above
that can be checked in code must be checked in code, with a failing exit
status when violated. The predecessor project declared coverage requirements
in prompt text, had no mechanism to enforce them, and silently produced
chapters at 6% of required coverage for weeks.

# TARGET HARDWARE
Samsung Galaxy S23 Ultra, Snapdragon 8 Gen 2, 8GB RAM with ~1.7GB
MemAvailable under normal load, 12GB zram. Adreno 740, Vulkan 1.3.
Practical local model ceiling: 3B at Q4. Phantom process monitor disabled and
Termux is battery-optimisation exempt, so long runs survive.
Secondary machine: laptop, i5-1135G7, ~2GB available RAM, Iris Xe. Weaker
than the phone. Used for review only.
