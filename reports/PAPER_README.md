# Prompt 14 paper package

`research_paper.md` is the main source and was written before PDF conversion.
`research_paper.pdf` is the professionally formatted research-style draft.
Companion files contain the relevant literature, methodology and results sections;
figure/table numbers are consistent with the main paper. New paper figures are
in `figures/research_paper/`. Existing experiment figures are preserved.

The draft covers the requested 18 items, counting its title, Abstract, Keywords
and 15 numbered sections. It includes eight figures and 13 tables with captions
and discussion, and an eight-entry verified bibliography. Authorship/affiliation
were not supplied and have not been invented. The paper is not asserted to be
publication-ready.

To regenerate from the supplied current artifacts in PowerShell, from the project root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-paper.txt
.\.venv\Scripts\python.exe -m scripts.generate_research_paper
.\.venv\Scripts\python.exe -m pytest -q tests/test_prompt14_paper.py
```

The generator uses Matplotlib's bundled DejaVu fonts, ReportLab and pypdf. It
recomputes aggregate metrics/confusion matrices from saved predictions, validates
dataset/model hashes and frozen selection, and audits the actual daily CSV.
It does not fit models, rerun tuning or change the application. Its layout was
rendered and inspected locally with Poppler; regeneration itself does not require
Poppler. Tests and PDF generation were verified on Linux, not executed on Windows.

The paper keeps the retrospective-test-exposure caveat. Additional audited limits
include six previously absent test cities (919 rows), clipping of calendar numerics,
only 38 Severe test examples, source AQI values beyond the official scale,
substantial missingness, absent weather and uncalibrated probabilities. None is
silently corrected by writing the report; a separate future experiment would be
needed to evaluate design changes.

`reference_verification.md`, `paper_data_audit.json` and `paper_verification.json`
record source checks and artifact-level verification. Cited foundational literature
predates 2023 where appropriate; recent dates are not fabricated. The primary
survey's publisher page and full 36-page paper were accessed.
