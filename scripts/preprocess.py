#!/usr/bin/env python3
"""
特徴キャッシュ生成スクリプト

メタデータからHuBERTラベルを事前計算して保存します。
これは学習前の準備ステップです。
"""

import argparse
from pathlib import Path

import torch
import yaml
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Preprocess datasets for StreamVC")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Config file path"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Process specific dataset only (default: all)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "valid"],
        help="Process specific split only (default: both)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available, else cpu)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)"
    )

    args = parser.parse_args()

    # 設定ファイルを読み込み
    with args.config.open("r") as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("StreamVC Feature Cache Generation")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Device: {args.device}")
    if args.max_samples:
        print(f"Max samples: {args.max_samples} (testing mode)")
    print()

    # データセット設定を取得
    data_config = config.get("data", {})
    datasets = data_config.get("datasets", [])
    cache_dir = Path(data_config.get("cache_dir", "data/cache"))
    sample_rate = data_config.get("sample_rate", 16000)
    sample_length_sec = data_config.get("sample_length_sec", 1.28)
    reference_length_sec = data_config.get("reference_length_sec", 1.0)
    hubert_cache = Path(data_config.get("hubert_cache", "data/cache/hubert"))

    # K-meansモデルのパス
    kmeans_path = hubert_cache / "km100.bin"

    if not kmeans_path.exists():
        print(f"❌ K-means model not found: {kmeans_path}")
        print("   Please download HuBERT models first:")
        print("   python scripts/download_datasets.py --datasets hubert")
        return

    print(f"K-means model: {kmeans_path}")
    print(f"Cache directory: {cache_dir}")
    print()

    # データセット処理
    from streamvc.data.preprocess import preprocess_metadata

    for dataset_cfg in datasets:
        dataset_name = dataset_cfg.get("name")

        # 特定のデータセットのみ処理する場合
        if args.dataset and dataset_name != args.dataset:
            continue

        metadata_path = Path(dataset_cfg.get("metadata"))
        dataset_root = Path(dataset_cfg.get("root"))

        print(f"Processing dataset: {dataset_name}")
        print(f"  Metadata: {metadata_path}")
        print(f"  Root: {dataset_root}")

        if not metadata_path.exists():
            print(f"  ⚠ Metadata not found, skipping...")
            continue

        # データセットルートの確認
        # LibriTTSの場合、実際の音声ファイルは data/libritts/LibriTTS/dev-clean/ にある
        if dataset_name == "libri_tts":
            # LibriTTS/dev-clean など実際の音声ディレクトリを探す
            possible_dirs = [
                dataset_root / "LibriTTS" / "dev-clean",
                dataset_root / "LibriTTS" / "train-clean-100",
                dataset_root / "LibriTTS" / "train-other-500",
            ]
            for possible_dir in possible_dirs:
                if possible_dir.exists():
                    dataset_root = possible_dir
                    print(f"  Using audio root: {dataset_root}")
                    break

        # メタデータの行数をカウント
        with metadata_path.open("r") as f:
            total_lines = sum(1 for _ in f)

        print(f"  Total entries: {total_lines}")

        # 分割ごとに処理
        splits_to_process = ["train", "valid"] if args.split is None else [args.split]

        for split in splits_to_process:
            print(f"\n  Processing {split} split...")

            # 既にキャッシュが存在するかチェック
            split_dir = cache_dir / dataset_name / split
            if split_dir.exists():
                existing_files = len(list(split_dir.glob("*.pt")))
                if existing_files > 0:
                    print(f"  ℹ Found {existing_files} existing cache files")
                    response = input(f"  Overwrite? (y/N): ").strip().lower()
                    if response != 'y':
                        print(f"  Skipping {split}...")
                        continue

            try:
                preprocess_metadata(
                    metadata_path=metadata_path,
                    cache_dir=cache_dir,
                    dataset_root=dataset_root,
                    dataset_name=dataset_name,
                    sample_rate=sample_rate,
                    sample_length_sec=sample_length_sec,
                    reference_length_sec=reference_length_sec,
                    kmeans_path=kmeans_path,
                    device=args.device,
                    split_filter=split,
                )

                # 処理されたファイル数を確認
                if split_dir.exists():
                    num_files = len(list(split_dir.glob("*.pt")))
                    print(f"  ✓ Generated {num_files} cache files for {split}")
                else:
                    print(f"  ⚠ No cache files generated for {split}")

            except Exception as e:
                print(f"  ❌ Error processing {split}: {e}")
                import traceback
                traceback.print_exc()
                continue

    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)
    print("\nVerify cache files:")
    for dataset_cfg in datasets:
        dataset_name = dataset_cfg.get("name")
        if args.dataset and dataset_name != args.dataset:
            continue
        for split in ["train", "valid"]:
            split_dir = cache_dir / dataset_name / split
            if split_dir.exists():
                num_files = len(list(split_dir.glob("*.pt")))
                print(f"  {dataset_name}/{split}: {num_files} files")

    print("\nNext step:")
    print("  python scripts/train.py --config configs/default.yaml --device cpu")


if __name__ == "__main__":
    main()
