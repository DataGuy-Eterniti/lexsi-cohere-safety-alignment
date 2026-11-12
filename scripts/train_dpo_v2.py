"""
Train the v2 LoRA DPO safety adapter on CohereLabs/tiny-aya-earth using Lexsi AlignTune.

Pulls the CuratorKIT-verified 80-pair dataset from Hugging Face Hub
(Eterniti/curated-cohere-safety-dpo), runs 2 epochs (160 steps) of DPO training,
copies data provenance into the output directory, and logs convergence metrics.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
from collections import Counter

from aligntune.core.backend_factory import create_rl_trainer
from huggingface_hub import HfApi, hf_hub_download, whoami

BASE_MODEL_ID = "CohereLabs/tiny-aya-earth"
DATASET_REPO_ID = "Eterniti/curated-cohere-safety-dpo"


def download_curated_dataset(data_dir: str) -> str:
    """Download dpo.jsonl and lexsi_provenance.json from Hugging Face Hub."""
    os.makedirs(data_dir, exist_ok=True)
    for fname in ("dpo.jsonl", "lexsi_provenance.json"):
        hf_hub_download(
            repo_id=DATASET_REPO_ID,
            repo_type="dataset",
            filename=fname,
            local_dir=data_dir,
        )

    jsonl_path = os.path.join(data_dir, "dpo.jsonl")
    with open(jsonl_path, "r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    print(f"Loaded {len(rows)} DPO pairs: {dict(Counter(r.get('category') for r in rows))}")
    return jsonl_path


def print_training_summary(output_dir: str) -> None:
    """Print loss, reward accuracy, and reward margin from trainer_state.json."""
    state_files = sorted(glob.glob(f"{output_dir}/**/trainer_state.json", recursive=True))
    if not state_files:
        return
    with open(state_files[-1], "r", encoding="utf-8") as handle:
        history = json.load(handle).get("log_history", [])
    for entry in history:
        if "loss" in entry:
            print(
                f"step={entry.get('step'):>3} | loss={entry.get('loss'):.4f} | "
                f"acc={entry.get('rewards/accuracies', 0):.3f} | "
                f"margin={entry.get('rewards/margins', 0):+.4f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="AlignTune DPO v2 trainer for Tiny Aya.")
    parser.add_argument("--data-dir", default="./curated_cohere_safety_data_v2")
    parser.add_argument("--output-dir", default="./aligned_cohere_adapter_v2")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--push-to-hub", action="store_true")
    args = parser.parse_args()

    jsonl_path = download_curated_dataset(args.data_dir)
    trainer = create_rl_trainer(
        model_name=BASE_MODEL_ID,
        dataset_name=jsonl_path,
        algorithm="dpo",
        backend="trl",
        num_epochs=args.epochs,
        batch_size=1,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_length,
        output_dir=args.output_dir,
    )
    trainer.train()

    shutil.copy(
        os.path.join(args.data_dir, "lexsi_provenance.json"),
        os.path.join(args.output_dir, "lexsi_provenance.json"),
    )
    print_training_summary(args.output_dir)

    if args.push_to_hub:
        api = HfApi()
        username = whoami()["name"]
        repo_id = f"{username}/tiny-aya-safety-dpo-adapter"
        api.upload_folder(
            folder_path=args.output_dir,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload AlignTune DPO v2 adapter (80 pairs, 2 epochs)",
        )
        print(f"Pushed adapter to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
