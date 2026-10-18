# Notice

The code in this repository is MIT-licensed (see LICENSE), with these exceptions, which stay under the
[Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/)
license of HERB (Salesforce AI Research, https://github.com/SalesforceAIResearch/HERB):

- The parts of these files ported from HERB's code, as marked in each file's docstring:
  `src/eval/grade.py` (HERB's `evaluate.py`: `normalize`, `f1_score_sets` and the judge prompts),
  `src/baselines/paper_tools.py` (`metadata_search_functions.py`), `src/baselines/paper_retriever.py`,
  `src/baselines/hybrid_once.py`, `src/baselines/react_paper.py` and `src/baselines/long_context.py`
  (`long_context_eval.py`).
- Every file derived from HERB data: the demo site's data under `site/data/`, the per-question tables in
  `results/tables/`, and the release assets (extracted facts, embedding cache, recorded run).

Raw HERB data is not in this repository; `make download` fetches it from HERB's repository at a pinned commit.
