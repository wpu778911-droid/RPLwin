# RPL FWT Action Chunking Experiments

This folder records the code entry points for two RPL four-wheel tracking
action-chunking experiments.

## Chunking

Result folder:

`results/RPL_FWTsim_chunk003_gatehard_long_v1`

Main code:

- `gops/env/env_FwFtracking/env_RPL_FWTsim.py`
- `gops/env/env_FwFtracking/env_RPL_FWTsim_chunk.py`
- `gops/env/env_FwFtracking/env_RPL_FWTsim_chunk003.py`
- `example_train/dsact/dsact_mlp_RPL_chunk.py`
- `gops/algorithm/dsact.py`

Training command:

```powershell
E:\StudyApp\Conda\envs\gops\python.exe example_train\dsact\dsact_mlp_RPL_chunk.py --env_id env_RPL_FWTsim_chunk003 --save_folder results/RPL_FWTsim_chunk003_gatehard_long_v1 --max_iteration 1500000 --seed 12345 --apprfunc_save_interval 50000
```

## Histstack Chunking

Result folder:

`results/RPL_FWTsim_chunk_histstack003_consistency_v1`

Main code:

- `gops/env/env_FwFtracking/env_RPL_FWTsim.py`
- `gops/env/env_FwFtracking/env_RPL_FWTsim_chunk.py`
- `gops/env/env_FwFtracking/env_RPL_FWTsim_chunk_histstack003.py`
- `example_train/dsact/dsact_mlp_RPL_chunk_histstack.py`
- `gops/algorithm/dsact.py`

Training command:

```powershell
E:\StudyApp\Conda\envs\gops\python.exe example_train\dsact\dsact_mlp_RPL_chunk_histstack.py --env_id env_RPL_FWTsim_chunk_histstack003 --save_folder results/RPL_FWTsim_chunk_histstack003_consistency_v1 --max_iteration 1500000 --seed 12345 --apprfunc_save_interval 50000
```

## DSACT Chunk TD Update

`gops/algorithm/dsact.py` supports macro transitions by reading
`next_discount_steps` from the replay buffer and using `gamma ** h` for the
bootstrap term.

`gops/env/env_FwFtracking/env_RPL_FWTsim_chunk.py` accumulates chunk rewards as
`sum gamma^i * r_{t+i}` and reports `discount_steps=h`, matching the h-step
action-chunking target.
