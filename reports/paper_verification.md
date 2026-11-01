# Prompt 14 verification

- The publisher page and the linked full 36-page primary survey were accessed and its bibliographic details checked. Additional citations use verified publisher, official documentation, government or author-preprint sources.
- The main Markdown source precedes PDF generation. All 18 requested items are present, counting the title, Abstract, Keywords and 15 numbered sections.
- The 17-page PDF reopens successfully and has 18 navigation bookmarks. All 17 pages were rendered and inspected. There are eight figures and 13 consecutively numbered tables, with captions and textual discussion. No extracted text lies outside the expected page bounds and no replacement glyphs were found.
- Actual aggregate metrics and confusion matrices were independently recomputed for all eight saved candidates from test_predictions.csv. Every candidate uses the same 4,421 ordered test identities. Model and dataset hashes match the saved experiment; the selected candidate is independently confirmed from validation scores.
- All 150 pre-existing project files outside the two requested report replacements (literature_review.md and references.bib) were compared byte-for-byte with the preceding project ZIP and remain unchanged. In particular, prompts 11-13 code, models and results were not modified or retrained.
- The full suite passed 156 tests, including five new paper-specific tests. Logs are in paper_verification/.

Outstanding limitations are stated in the paper rather than concealed: historical test exposure, mixed temporal/city expansion, calendar clipping, missingness and unit/measurement provenance, extreme source AQI values, limited Severe support, small tuning/interpretation budgets, absent weather, uncalibrated probabilities and missing prospective validation. The generated PDF is a research-style project draft, not proof of publication readiness. Authorship and affiliation were not supplied and remain unassigned.
