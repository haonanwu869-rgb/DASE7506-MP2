# Experiment evidence index

Final model: `../model/`. Final official scores: `../evaluation/`. Historical absolute paths are provenance, not inference dependencies.

| Directory | Purpose |
|---|---|
| four_steps | Baseline, logistic regression, three MLP losses, relative features, attention, trees |
| stability | Five fixed training seeds for original attention |
| reliability_draft_v1 | First checker version, superseded after implementation review |
| reliability_draft_v2 | Conservative checker revision, superseded |
| reliability | Final reliability features; ablations and five seeds |
| ensembles | Four fixed equal-probability ensembles and single-model controls |
| source_snapshot | Development source at packaging time; final entry points are in ../code |

Each results.json includes scores, resource measurements and checkpoint hashes. Some original working notes are Chinese. These records intentionally preserve the state at creation; statements that test was unused describe the development stage, not the current final release. Figures use stability/results.json, reliability/results.json and ensembles/results.json only. No new fitting or test selection is involved.
