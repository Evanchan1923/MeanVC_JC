# Project Context

- MeanVC downloaded checkpoints are expected under `/srv/scratch/speechdata/SAPC_Team/meanVC_checkpoint`.
- SAPC Severe Hugging Face dataset root is `/srv/scratch/speechdata/speech-corpora/dysarthric/SAPC_HF/SAPC_Severe`, with `train` used for fine-tuning and `dev` reserved for later validation.
- `configs/speaker_research_nutts_gt200.csv` is derived from `configs/speaker_research_all.csv` using `n_utts > 200` and descending `n_utts` order.
