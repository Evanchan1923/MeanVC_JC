from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio.compliance.kaldi as kaldi
import yaml
from datasets import Audio, DatasetDict, load_from_disk
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.verification import init_model as init_sv_model
from src.preprocess.extrace_mel_10ms import MelSpectrogramFeatures


PLACEHOLDER_RE = re.compile(r"\$\{([^}]+)\}")


def load_yaml_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return resolve_placeholders(cfg)


def resolve_placeholders(cfg: dict[str, Any]) -> dict[str, Any]:
    for _ in range(10):
        resolved = _resolve_value(cfg, cfg)
        if resolved == cfg:
            return resolved
        cfg = resolved
    raise ValueError("YAML placeholder resolution exceeded 10 passes")


def _resolve_value(value: Any, root: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {k: _resolve_value(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_value(v, root) for v in value]
    if not isinstance(value, str):
        return value

    full_match = PLACEHOLDER_RE.fullmatch(value)
    if full_match:
        return _resolve_value(_lookup(root, full_match.group(1)), root)

    def replace(match: re.Match[str]) -> str:
        return str(_resolve_value(_lookup(root, match.group(1)), root))

    return PLACEHOLDER_RE.sub(replace, value)


def _lookup(root: dict[str, Any], dotted_key: str) -> Any:
    value: Any = root
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"Unknown YAML placeholder: {dotted_key}")
        value = value[part]
    return value


def to_path(value: str | Path, repo_root: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute() or repo_root is None:
        return path
    return repo_root / path


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def first_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def safe_name(value: Any, fallback: str) -> str:
    text = str(value or fallback).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("._-") or fallback


def load_hf_split(dataset_path: Path, split: str, audio_column: str):
    loaded = load_from_disk(str(dataset_path))
    if isinstance(loaded, DatasetDict):
        if split not in loaded:
            raise KeyError(f"Split '{split}' not found in {dataset_path}; available splits: {list(loaded.keys())}")
        dataset = loaded[split]
    else:
        dataset = loaded

    if audio_column in dataset.features:
        dataset = dataset.cast_column(audio_column, Audio(decode=False))
    return dataset


def decode_audio_bytes(row: dict[str, Any], audio_column: str, fallback_path_column: str | None, dataset_path: Path) -> tuple[np.ndarray, int]:
    audio = row.get(audio_column)
    audio_bytes = None
    audio_path = None

    if isinstance(audio, dict):
        audio_bytes = audio.get("bytes")
        audio_path = audio.get("path")
    elif isinstance(audio, (bytes, bytearray)):
        audio_bytes = audio

    if audio_bytes is not None:
        wav, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    else:
        if not audio_path and fallback_path_column:
            audio_path = row.get(fallback_path_column)
        if not audio_path:
            raise ValueError(f"No audio bytes or path found in column '{audio_column}'")
        wav, sr = sf.read(resolve_audio_path(str(audio_path), dataset_path), dtype="float32", always_2d=False)

    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = np.asarray(wav, dtype=np.float32)
    return wav, int(sr)


def resolve_audio_path(audio_path: str, dataset_path: Path) -> str:
    path = Path(audio_path).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.append(dataset_path / path)
        candidates.append(dataset_path.parent / path)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"Audio path does not exist: {audio_path}")


def resample_to_target(wav: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return np.asarray(wav, dtype=np.float32)


def extract_fbanks(wav: np.ndarray, sample_rate: int) -> torch.Tensor:
    wav_tensor = torch.from_numpy(wav * (1 << 15)).unsqueeze(0)
    fbanks = kaldi.fbank(
        wav_tensor,
        frame_length=25,
        frame_shift=10,
        snip_edges=True,
        num_mel_bins=80,
        energy_floor=0.0,
        dither=0.0,
        sample_frequency=sample_rate,
    )
    return fbanks.unsqueeze(0)


def bn_mode_params(mode: str) -> tuple[int, int, int]:
    if mode == "200ms":
        return 5, 20, 23
    if mode == "160ms":
        return 4, 16, 19
    raise ValueError(f"Unsupported bn_frame_mode '{mode}'. Use '200ms' or '160ms'.")


def extract_bn_from_audio(asr_model, wav: np.ndarray, sample_rate: int, mode: str, device: torch.device) -> np.ndarray:
    decoding_chunk_size, step, window = bn_mode_params(mode)
    fbanks = extract_fbanks(wav, sample_rate).float().to(device)
    offset = 0
    required_cache_size = decoding_chunk_size * 2
    att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device="cpu")
    cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device="cpu")
    bns = []

    with torch.no_grad():
        for i in range(0, fbanks.shape[1], step):
            fbank = fbanks[:, i:i + window, :]
            if fbank.shape[1] < 10:
                break
            encoder_output, att_cache, cnn_cache = asr_model.forward_encoder_chunk(
                fbank,
                offset,
                required_cache_size,
                att_cache,
                cnn_cache,
            )
            bns.append(encoder_output)
            offset += encoder_output.size(1)

    if not bns:
        raise ValueError("ASR encoder produced no BN frames")
    return torch.cat(bns, dim=1).squeeze(0).detach().cpu().numpy()


def extract_mel_from_audio(mel_extractor: MelSpectrogramFeatures, wav: np.ndarray, device: torch.device) -> np.ndarray:
    wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).to(device)
    with torch.no_grad():
        mel = mel_extractor(wav_tensor)
    return mel.squeeze(0).detach().cpu().numpy().T


def extract_speaker_embedding(sv_model, wav: np.ndarray, device: torch.device) -> np.ndarray:
    wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).to(device)
    with torch.no_grad():
        emb = sv_model(wav_tensor)
    return emb.squeeze(0).detach().cpu().numpy()


class FeatureModels:
    def __init__(self, cfg: dict[str, Any], device: torch.device):
        self.cfg = cfg
        self.device = device
        self.asr_model = None
        self.mel_extractor = None
        self.sv_model = None

    def load(self):
        if self.asr_model is not None:
            return

        asr_ckpt_path = Path(self.cfg["asr_ckpt_path"]).expanduser()
        sv_ckpt_path = Path(self.cfg["speaker_verification_ckpt_path"]).expanduser()
        if not asr_ckpt_path.exists():
            raise FileNotFoundError(f"ASR checkpoint not found: {asr_ckpt_path}")
        if not sv_ckpt_path.exists():
            raise FileNotFoundError(f"Speaker verification checkpoint not found: {sv_ckpt_path}")

        self.asr_model = torch.jit.load(str(asr_ckpt_path), map_location=self.device).to(self.device)
        self.asr_model.eval()
        self.mel_extractor = MelSpectrogramFeatures(sample_rate=int(self.cfg["sample_rate"])).to(self.device)
        self.mel_extractor.eval()
        self.sv_model = init_sv_model(self.cfg.get("speaker_embedding_model", "wavlm_large"), str(sv_ckpt_path))
        self.sv_model.eval()
        self.sv_model.to(self.device)


def prepare_features(cfg: dict[str, Any], repo_root: Path, run_dir: Path, output_dir: Path) -> Path:
    user_cfg = cfg["user_settings"]
    prepare_cfg = cfg["prepare"]
    dataset_path = to_path(user_cfg["hf_dataset_path"])
    split = str(user_cfg.get("train_split", "train"))
    audio_column = prepare_cfg.get("audio_column", "audio")
    id_column = prepare_cfg.get("id_column", "id")
    speaker_column = prepare_cfg.get("speaker_column", "speaker")
    duration_column = prepare_cfg.get("duration_column", "duration")
    fallback_audio_path_column = prepare_cfg.get("fallback_audio_path_column", "audio_filepath")
    sample_rate = int(prepare_cfg.get("sample_rate", 16000))
    output_subdir = prepare_cfg.get("output_subdir", "prepared_train")
    force = bool_value(prepare_cfg.get("force", False))
    seed = int(prepare_cfg.get("seed", 42))
    limit = prepare_cfg.get("limit")
    max_duration = prepare_cfg.get("max_duration")
    min_duration = float(prepare_cfg.get("min_duration", 0.05))
    max_prompts_per_utt = int(prepare_cfg.get("max_prompt_mels_per_utt", 8))
    bn_frame_mode = prepare_cfg.get("bn_frame_mode", "200ms")

    prepared_dir = output_dir / output_subdir
    bn_dir = prepared_dir / "bn"
    mel_dir = prepared_dir / "mel"
    xvector_dir = prepared_dir / "xvector"
    for directory in (bn_dir, mel_dir, xvector_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manifest_path = prepared_dir / "train_manifest.txt"
    metadata_path = prepared_dir / "metadata.jsonl"
    failure_path = prepared_dir / "failures.jsonl"

    dataset = load_hf_split(dataset_path, split, audio_column)
    total = len(dataset) if limit is None else min(int(limit), len(dataset))
    random.seed(seed)
    np.random.seed(seed)
    device = torch.device(prepare_cfg.get("feature_device", "cuda:0") if torch.cuda.is_available() else "cpu")
    feature_models = FeatureModels(prepare_cfg, device)
    records: list[dict[str, Any]] = []
    failures = 0
    max_errors = int(prepare_cfg.get("max_errors", 100))

    with metadata_path.open("w", encoding="utf-8") as metadata_f, failure_path.open("w", encoding="utf-8") as failure_f:
        for index in tqdm(range(total), desc=f"Preparing {split} features"):
            row = dict(dataset[index])
            duration = row.get(duration_column)
            if duration is not None:
                duration_f = float(duration)
                if duration_f < min_duration:
                    continue
                if max_duration is not None and duration_f > float(max_duration):
                    continue

            utt = safe_name(row.get(id_column), f"utt_{index:08d}")
            speaker = safe_name(row.get(speaker_column), "unknown_speaker")
            stem = f"{index:08d}_{utt}"
            bn_path = bn_dir / f"{stem}.npy"
            mel_path = mel_dir / f"{stem}.npy"
            xvector_path = xvector_dir / f"{stem}.npy"

            try:
                if force or not (bn_path.exists() and mel_path.exists() and xvector_path.exists()):
                    feature_models.load()
                    wav, sr = decode_audio_bytes(row, audio_column, fallback_audio_path_column, dataset_path)
                    wav = resample_to_target(wav, sr, sample_rate)
                    if wav.size == 0:
                        raise ValueError("Decoded audio is empty")
                    bn = extract_bn_from_audio(feature_models.asr_model, wav, sample_rate, bn_frame_mode, device)
                    mel = extract_mel_from_audio(feature_models.mel_extractor, wav, device)
                    emb = extract_speaker_embedding(feature_models.sv_model, wav, device)
                    np.save(bn_path, bn)
                    np.save(mel_path, mel)
                    np.save(xvector_path, emb)

                record = {
                    "id": utt,
                    "speaker": speaker,
                    "bn_path": str(bn_path),
                    "mel_path": str(mel_path),
                    "xvector_path": str(xvector_path),
                }
                records.append(record)
                metadata_f.write(json.dumps({**record, "source_index": index}, ensure_ascii=False) + "\n")
            except Exception as exc:
                failures += 1
                failure_f.write(json.dumps({"source_index": index, "id": row.get(id_column), "error": repr(exc)}, ensure_ascii=False) + "\n")
                if failures > max_errors:
                    raise RuntimeError(f"Exceeded max_errors={max_errors}; see {failure_path}") from exc

    if not records:
        raise RuntimeError("No training records were prepared")

    by_speaker: dict[str, list[str]] = {}
    for record in records:
        by_speaker.setdefault(record["speaker"], []).append(record["mel_path"])

    with manifest_path.open("w", encoding="utf-8") as manifest_f:
        for record in records:
            prompt_paths = [path for path in by_speaker[record["speaker"]] if path != record["mel_path"]]
            if not prompt_paths:
                prompt_paths = [record["mel_path"]]
            random.shuffle(prompt_paths)
            prompt_paths = prompt_paths[:max_prompts_per_utt]
            parts = [record["id"], record["bn_path"], record["mel_path"], record["xvector_path"], *prompt_paths]
            manifest_f.write("|".join(parts) + "\n")

    summary = {
        "dataset_path": str(dataset_path),
        "split": split,
        "records": len(records),
        "failures": failures,
        "manifest_path": str(manifest_path),
        "run_dir": str(run_dir),
    }
    with (prepared_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[INFO] Prepared {len(records)} records with {failures} failures")
    print(f"[INFO] Manifest: {manifest_path}")
    return manifest_path


def command_arg(name: str) -> str:
    return "--" + name.replace("_", "-")


def train_args_from_config(train_cfg: dict[str, Any], manifest_path: Path, run_dir: Path, output_dir: Path) -> dict[str, Any]:
    args = dict(train_cfg.get("args", {}))
    args["dataset_path"] = str(manifest_path)
    args["save_dir"] = str(output_dir / "checkpoints")
    args.setdefault("exp_name", run_dir.name)
    return args


def run_training(cfg: dict[str, Any], repo_root: Path, run_dir: Path, output_dir: Path, manifest_path: Path) -> None:
    user_cfg = cfg["user_settings"]
    train_cfg = cfg["train"]
    devices = int(user_cfg.get("devices", 1))
    gpu_ids = train_cfg.get("gpu_ids")
    if gpu_ids is None:
        gpu_ids = ",".join(str(i) for i in range(devices))

    train_script = to_path(train_cfg.get("script", "src/train/train.py"), repo_root)
    accelerate_config = to_path(train_cfg.get("accelerate_config", "default_config.yaml"), repo_root)
    train_args = train_args_from_config(train_cfg, manifest_path, run_dir, output_dir)
    port = int(train_cfg.get("main_process_port") or first_open_port())

    command = [
        "accelerate",
        "launch",
        "--config-file",
        str(accelerate_config),
        "--main_process_port",
        str(port),
        "--num_processes",
        str(devices),
        "--gpu_ids",
        str(gpu_ids),
        str(train_script),
    ]
    for key, value in train_args.items():
        if value is None:
            continue
        command.extend([command_arg(key), str(value)])

    env = os.environ.copy()
    for key, value in cfg.get("environment", {}).get("env", {}).items():
        env[str(key)] = str(value)
    env["PYTHONPATH"] = f"{repo_root}:{env.get('PYTHONPATH', '')}".rstrip(":")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Launching MeanVC fine-tuning:")
    print(" ".join(command))
    subprocess.run(command, cwd=str(repo_root), env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare HF bytes audio and fine-tune MeanVC.")
    parser.add_argument("--config", required=True, help="Path to MeanVC fine-tuning YAML config.")
    parser.add_argument("--prepare-only", action="store_true", help="Only prepare HF dataset features and manifest.")
    parser.add_argument("--train-only", action="store_true", help="Skip feature preparation and train from the existing manifest.")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    cfg = load_yaml_config(config_path)
    repo_root = to_path(cfg["user_settings"].get("repo_root", "."), Path.cwd()).resolve()
    run_root = to_path(cfg["run"]["run_root"]).resolve()
    run_name = safe_name(cfg["run"]["name"], "meanvc_finetune")
    run_dir = run_root / run_name
    output_dir = run_dir / cfg["run"].get("output_subdir", "artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    stages = cfg.get("pipeline", {}).get("stages", ["prepare", "train"])
    manifest_path = output_dir / cfg["prepare"].get("output_subdir", "prepared_train") / "train_manifest.txt"
    if "prepare" in stages and not args.train_only:
        manifest_path = prepare_features(cfg, repo_root, run_dir, output_dir)
    elif not manifest_path.exists():
        raise FileNotFoundError(f"Manifest does not exist for train-only mode: {manifest_path}")

    if args.prepare_only:
        return

    if "train" in stages:
        run_training(cfg, repo_root, run_dir, output_dir, manifest_path)


if __name__ == "__main__":
    main()
