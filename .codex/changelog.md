# Changelog

## 2026-08-13

- Added MeanVC SAPC Severe fine-tuning pipeline files and training defaults.
- Added GPU Accelerate and PBS launcher configs for server fine-tuning.
- Fixed MeanVC trainer manifest loading and made hardcoded validation opt-in.
- Renamed the fine-tune config and PBS launcher to `meanVC_ft_v1` and updated MeanVC checkpoint paths to `/srv/scratch/speechdata/SAPC_Team/meanVC_checkpoint`.
- Moved `meanVC_ft_v1.pbs` to the repository root.
- Updated speaker-verification checkpoint defaults to use the shared `meanVC_checkpoint` directory.
- Removed obsolete `example.pbs` and `example.yaml`.
- Documented the recommendation to keep shared multi-speaker MeanVC fine-tuning as the default and use per-speaker runs selectively.
- Added filtered speaker research CSV for speakers with more than 200 utterances, sorted by utterance count descending.

## 2026-06-30
