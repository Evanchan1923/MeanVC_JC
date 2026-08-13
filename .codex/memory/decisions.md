# Decisions

- The MeanVC SAPC fine-tuning pipeline prepares the Hugging Face dataset into the existing MeanVC manifest format instead of replacing `DiffusionDataset` and `Trainer`.
- HF audio is read by casting the `Audio` column to `decode=False` and decoding the stored bytes with `soundfile`; `torchaudio` and `torchcodec` are not used for audio I/O.
- Validation is opt-in through `run_validation` because the existing trainer validation uses hardcoded paths and the current task focuses on fine-tuning only.
- The fine-tune config and PBS launcher are named `meanVC_ft_v1` for the first server fine-tuning version.
