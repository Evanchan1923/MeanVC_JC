# training script.

from pathlib import Path

from src.model import DiT, Trainer
from src.model.utils import load_checkpoint

from prefigure.prefigure import get_all_args
import json
import os

import time

os.environ['OMP_NUM_THREADS']="1"
os.environ['MKL_NUM_THREADS']="1"


def _is_empty_arg(value):
    if value is None:
        return True
    return str(value).strip().strip("\"'") == ""


def _checkpoint_path(args):
    if not _is_empty_arg(getattr(args, "save_dir", "")):
        return str(Path(args.save_dir).expanduser())
    return str(Path(__file__).resolve().parents[2] / "ckpts" / args.exp_name)


def _has_resume_checkpoint(checkpoint_path):
    checkpoint_dir = Path(checkpoint_path).expanduser()
    return checkpoint_dir.exists() and any(path.suffix == ".pt" for path in checkpoint_dir.iterdir())


def main():
    args = get_all_args()
    print("Parsed arguments:", args)
    if isinstance(args.feature_list, str):
        args.feature_list = args.feature_list.split()
    if isinstance(args.additional_feature_list, str):
        args.additional_feature_list = args.additional_feature_list.split()
    if isinstance(args.feature_pad_values, str):
        args.feature_pad_values = args.feature_pad_values.split() 

    with open(args.model_config) as f:
        model_config = json.load(f)

    if model_config["model_type"] == "DiT":
        wandb_resume_id = None
        model_cls = DiT

    
    model = DiT(**model_config["model"])

    total_params = sum(p.numel() for p in model.parameters())/ 1000000
    print("Total parameters: {:.6f} M".format(total_params))

    checkpoint_path = _checkpoint_path(args)
    pretrained_ckpt_path = getattr(args, "pretrained_ckpt_path", "")
    if not _is_empty_arg(pretrained_ckpt_path) and not _has_resume_checkpoint(checkpoint_path):
        print(f"Loading pretrained checkpoint: {pretrained_ckpt_path}")
        model = load_checkpoint(model, pretrained_ckpt_path, device="cpu", use_ema=True)
    elif _has_resume_checkpoint(checkpoint_path):
        print(f"Found existing training checkpoint in {checkpoint_path}; trainer resume will take precedence.")

    print(args.num_warmup_updates)
    trainer = Trainer(
        model,
        args,
        args.epochs,
        args.learning_rate,
        num_warmup_updates=args.num_warmup_updates,
        save_per_updates=args.save_per_updates,
        checkpoint_path=checkpoint_path,
        grad_accumulation_steps=args.grad_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        wandb_project="meanvc",
        wandb_run_name=args.exp_name,
        wandb_resume_id=wandb_resume_id,
        last_per_steps=args.last_per_steps,
        bnb_optimizer=False,
        reset_lr=args.reset_lr,
        batch_size=args.batch_size,
        grad_ckpt=args.grad_ckpt
    )

    trainer.train(
        resumable_with_seed=args.resumable_with_seed,  # seed for shuffling dataset
    )


if __name__ == "__main__":
    main()
