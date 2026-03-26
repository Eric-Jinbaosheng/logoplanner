# LoGoPlanner

This repository contains a reconstructed training and evaluation codebase for LoGoPlanner, based on the public LoGoPlanner release, with:

- end-to-end pointgoal training code
- LaCT-based decoder replacement experiments
- pointgoal evaluation scripts

Main code lives in `baselines/logoplanner/`.

Additional top-level files included here:

- `eval_pointgoal_wheeled.py`
- `eval_startgoal_wheeled.py`
- `eval_nogoal_wheeled.py`
- `eval_imagegoal_wheeled.py`
- `configs/tasks/wheeled_task.py`

Training-related additions include:

- `baselines/logoplanner/dataset_interndata_n1.py`
- `baselines/logoplanner/train_logoplanner_policy.py`
- `baselines/logoplanner/validate_trained_logoplanner_checkpoint.py`
- `baselines/logoplanner/checkpoint_utils.py`

LaCT-related code lives in:

- `baselines/logoplanner/lact_decoder.py`

This repository is intended as a research code release for reproducing and extending LoGoPlanner-style pointgoal training experiments.
