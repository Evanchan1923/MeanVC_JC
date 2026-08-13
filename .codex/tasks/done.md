# Done

- 2026-08-13: Added a MeanVC fine-tuning pipeline for the SAPC Severe Hugging Face dataset, including HF bytes-audio feature preparation, config YAML, PBS launcher, trainer fixes, and static checks.
- 2026-08-13: Renamed the MeanVC fine-tune YAML/PBS files to `meanVC_ft_v1` and updated MeanVC checkpoint paths to `/srv/scratch/speechdata/SAPC_Team/meanVC_checkpoint`.
- 2026-08-13: Moved `meanVC_ft_v1.pbs` to the repository root for direct PBS submission.
- 2026-08-13: Updated all speaker-verification checkpoint defaults to use `/srv/scratch/speechdata/SAPC_Team/meanVC_checkpoint/wavlm_large_finetune.pth`.
- 2026-08-13: Removed obsolete `example.pbs` and `example.yaml` after adding `meanVC_ft_v1` files.
