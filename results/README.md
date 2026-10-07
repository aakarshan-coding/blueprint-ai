# Results

Every benchmark run the decisions log cites, as the JSON the benchmark wrote. One file
per run; a five-run experiment is five files. Summarise any set with:

    python -m graphrag.eval.summarize_runs results/exp9/*.json

| folder | what it measured | decision |
|---|---|---|
| `early/` | single runs from before the noise floor was known (run1 was a known-buggy build, run5 an inflated grader) | D45–D56 |
| `repeat/` | the first five-run protocol on identical code: the noise floor | D60 |
| `exp1/`, `exp1b/` | the shared-synthesis-prompt confound, then the refusal wording restored | D61, D62 |
| `exp2/` | an abandoned partial experiment (two runs) | D63 |
| `exp3/` | the D63–D65 batch on 90 questions | D66 |
| `exp4/` | the D66–D69 batch (same-module rung, overrides, provenance) | D70 |
| `exp5/` | the D71–D74 batch (dedupe bug, last hop, anchors) | D75 |
| `exp5_regraded/` | exp5 re-scored under the audited grader | D79 |
| `exp6/` | the D76–D77 batch (forward call paths, grounded mentions) | D78 |
| `exp6_regraded/` | exp6 re-scored under the audited grader | D79 |
| `exp7/` | the D80–D84 batch (jedi goto, counting, anchors, new edge types) | D85 |
| `exp7_scored/` | exp7 re-scored with partial credit and the report-based judge | D87 |
| `exp8/`, `exp8_scored/` | the 220-question set, as recorded and after two reference fixes | D89 |
| `exp9/` | the D90 batch on 220 questions: the current table | D91 |
| `oracle/` | the runtime oracle: a `sys.settrace` trace of the requests test suite and the graph scored against it | D67, D80 |
| `stability/` | grader-variance and retrieval-stability probes | D51, D53 |

The numbers in the README come from `exp9/`. "regraded"/"scored" folders are the same
answers under a later grader; nothing was re-generated.
