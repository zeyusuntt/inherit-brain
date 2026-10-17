# Dev acceptance audit, round 6

Model-based dev acceptance. Two model auditors, a third pass for disagreements. Agreement between models does not establish equivalence to human labels. No source text appears in this report.

- complete: True (third passes still pending: 0)
- sample hash: 34251f49d0d932fd6ba18e6f3a54e27b85283899a676f25d7f38f7c73d98f359
- prompt version: audit-prompts/v5
- verdict hashes: {'codex': '5eb20ff33b604d0035626a142f19178b6bb1e853237daccf30ec743d0b8da025', 'claude': '69e2f7cc6fc804dabdc3ad7a52bdf7c1b07470febb796ec74f438be88d373e69', 'third': 'd0accf5caef913ccd5450deba00c6674e11c900542fec2c00965414aec3e7736'}
- integrity problems: none

## Prompt changes

This round uses audit-prompts/v5; round 1 used audit-prompts/v1 and round 2 audit-prompts/v2 (both kept unchanged). Changes in this version:
- review_relation and recall_reviewer: product identity is defined (records in the product's own file, including the names its own channels use for it, belong to it; another company's product does not)
- review_relation and name_resolution: the name-candidate rule text is the implemented rule (same-name attendees first; non-attendees only when no attendee exists or when the speaker's role cue matches a non-attendee and no attendee; 3 x role cue + 2 x related post + 1 x mentioned; unique top score resolves, ties ambiguous)

## Method

Each packet is judged on its own and answered per packet. Packets are scheduled in batches for the fan-out: a batch holds only packets of one kind and the rubric, at most 10 packets and about 120000 characters (a larger packet goes alone). The third pass shares a model family with auditor 1 (Codex).

## Auditors

- codex: model gpt-6.1-sol, version codex-cli 0.162.1, settings codex exec -s read-only, model_reasoning_effort=high, one fresh process per batch started in an empty directory holding only the batch file; one retry when the reply is not JSON, prompt_hash fce3310b100d74dd845bf22ee06cff4e67540adb2877ee2b4662063ba28d422b
- claude: model claude-sonnet-5-5, version Claude Code workflow agent (model alias sonnet), settings one fresh agent per batch, two at a time; reads only the batch file and writes only its verdict file; no other tools or files, prompt_hash 5265542fbc58c8e776f039daea52b4e4311242568e9cc1272007895a82c2bea7
- third: model gpt-6.1-sol, version codex-cli 0.162.1, settings codex exec -s read-only, model_reasoning_effort=high, one fresh process per batch started in an empty directory holding only the batch file; one retry when the reply is not JSON; third pass over third_batches (shares the model family with auditor 1), prompt_hash fce3310b100d74dd845bf22ee06cff4e67540adb2877ee2b4662063ba28d422b

Pools smaller than the quota (census, no padding): whatif_path 0/20

## Precision (adjudicated)

| category | correct/n | precision | double-positive | agreement | kappa | to third | unresolved |
|---|---|---|---|---|---|---|---|
| bug_event | 50/50 | 1.0 | 50/50 | 50/50 | None | 0 | 0 |
| name_resolution | 50/50 | 1.0 | 50/50 | 50/50 | None | 0 | 0 |
| review_relation | 50/50 | 1.0 | 50/50 | 50/50 | None | 0 | 0 |
| whatif_edge | 29/30 | 0.9667 | 29/30 | 30/30 | 1.0 | 0 | 0 |
| whatif_unowned | 30/30 | 1.0 | 30/30 | 30/30 | None | 0 | 0 |

## Recall

| kind | matched/truth | recall | packets | auditor lists equal |
|---|---|---|---|---|
| recall_bug | 75/76 | 0.9868 | 3 | 3 |
| recall_reviewer | 42/48 | 0.875 | 3 | 1 |

Reviewer recall alongside the gate: strict per-document recall 42/48 (an item that names a version is matched against that version only); distinct-reviewer coverage 38/43 people.

Disclosure: After inspecting dev rounds 1-4, the reviewer-recall endpoint was amended: targets are document types (all versions), author acknowledgements are excluded by act, and strict per-document recall and distinct-reviewer coverage are reported alongside. Earlier rounds' results are unchanged and remain failed.

Not estimable (empty dev pool, excluded from pass and fail): whatif_path.

Disclosure: 0 first-report messages carrying more than one bug in the extraction are in the bug recall packets; those bugs share one identity key in the comparison.
Disclosure: later sessions that refer to a bug only by anaphora ("this one"), with no title, customer company or PR link, are not in the bug precision packets.

## Prompt hashes

- common_precision: 51da091166f516ba0cc9ad46dc4c742421a34e4f9950478b3c8b749db9f7808c
- common_recall: 0e3480c20d3c7bb9c172379cad668dad0513ff4771731956f064d1939b9ea926
- open_work: 449aacfb21be7e39bdb3df08fcfaf8a5a015fdede550d84a0b9e05809d528359
- recall_rules:recall_bug: 673756c2fa5f50cabda32a9e6cc21085f0c4a7a43a58c1abd21ddfbe0415896f
- recall_rules:recall_reviewer: cbe6294c4d1aaeb2b2d8049d1358a320ffd45e919ea72faeade9e6b47469c5bb
- rubric:bug_event: 371137e11726258ab47bcf9e143b5c71883ecf806fdf0df404229a004d0700ba
- rubric:name_resolution: 02abe5c837f2fab89cda5402407964d0177f24987b09429a60cc8a026c261c09
- rubric:review_relation: 2c5bb5ee41266f8b48721f7f5a651216ea0f76b431d773ffbe791a8720eb1e6b
- rubric:whatif_edge: ac12e5062905da7a6717e7a2b71b2ec98edafbf857b7282f61b8ce927ee7d29b
- rubric:whatif_path: 82ab78116e7ec3c122e7f49fd02de45467dad9694e5838aeccca6c15c118703f
- rubric:whatif_unowned: e7e3e2621b2a77cb2979006765349d48cfbc756f3d342883b858c9c9ad912f45
- third_precision: df0b74dafd6300b8793d02debb0fa81fcce869336097f051ff3dbb0c8fdaf6e2
- third_recall: 31c858f27f902632a6750b5a1dc103c09bd3d5be97430f64ea3958ded14c1bdc
