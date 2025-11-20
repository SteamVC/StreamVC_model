#!/usr/bin/env python3
"""
メタデータ生成スクリプト

各データセットから音声ファイルをスキャンしてmetadata.jsonlを生成します。
各エントリには以下の情報が含まれます:
- id: 一意の識別子
- split: train/valid
- source: ソース音声ファイルパス
- reference: 参照音声ファイルパス（話者埋め込み用）
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

import soundfile as sf
from tqdm import tqdm


def scan_libritts(
    data_dir: Path,
    output_path: Path,
    valid_ratio: float = 0.05,
    max_speakers: int = None,
    max_utts_per_speaker: int = None,
) -> Tuple[int, int]:
    """
    LibriTTSデータセットをスキャンしてメタデータを生成

    Args:
        data_dir: LibriTTSのルートディレクトリ (data/libritts/LibriTTS/dev-clean)
        output_path: 出力先のmetadata.jsonlパス
        valid_ratio: 検証セットの割合
        max_speakers: 最大話者数 (Noneなら全話者)
        max_utts_per_speaker: 話者あたりの最大発話数 (Noneなら全発話)

    Returns:
        (train_count, valid_count): 生成したエントリ数
    """
    print(f"\n=== Scanning LibriTTS: {data_dir} ===")

    if not data_dir.exists():
        print(f"⚠ Directory not found: {data_dir}")
        return 0, 0

    # 話者ディレクトリを取得
    speaker_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    if max_speakers:
        speaker_dirs = speaker_dirs[:max_speakers]

    print(f"Found {len(speaker_dirs)} speakers")

    # 全話者の音声ファイルを収集
    speaker_files = {}
    for speaker_dir in tqdm(speaker_dirs, desc="Collecting speaker files"):
        speaker_id = speaker_dir.name
        wav_files = sorted(speaker_dir.rglob("*.wav"))

        if not wav_files:
            continue

        if max_utts_per_speaker:
            wav_files = wav_files[:max_utts_per_speaker]

        speaker_files[speaker_id] = wav_files

    if len(speaker_files) == 0:
        print(f"⚠ Warning: No speakers found.")
        return 0, 0

    train_entries = []
    valid_entries = []

    for speaker_id, wav_files in tqdm(speaker_files.items(), desc="Processing speakers"):
        # 検証セット用にランダムサンプリング
        num_valid = max(1, int(len(wav_files) * valid_ratio))
        valid_indices = set(random.sample(range(len(wav_files)), num_valid))

        for i, wav_file in enumerate(wav_files):
            # 参照音声は同一話者の別発話を使用（StreamVC論文の自己再構成学習）
            if len(wav_files) > 1:
                ref_idx = (i + 1) % len(wav_files)
                ref_file = wav_files[ref_idx]
            else:
                ref_file = wav_file  # 1発話しかない場合は同じファイル

            # データディレクトリからの相対パスを取得
            source_rel = wav_file.relative_to(data_dir)
            ref_rel = ref_file.relative_to(data_dir)

            entry = {
                "id": f"{speaker_id}_{wav_file.stem}",
                "split": "valid" if i in valid_indices else "train",
                "source": str(source_rel),
                "reference": str(ref_rel),
                "speaker_id": speaker_id,
            }

            if i in valid_indices:
                valid_entries.append(entry)
            else:
                train_entries.append(entry)

    # メタデータを保存
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for entry in train_entries + valid_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"✓ Generated metadata: {output_path}")
    print(f"  Train: {len(train_entries)} entries")
    print(f"  Valid: {len(valid_entries)} entries")

    return len(train_entries), len(valid_entries)


def scan_vctk(
    data_dir: Path,
    output_path: Path,
    valid_ratio: float = 0.05,
    max_speakers: int = None,
    max_utts_per_speaker: int = None,
) -> Tuple[int, int]:
    """
    VCTKデータセットをスキャンしてメタデータを生成

    Args:
        data_dir: VCTKのwav48_silence_trimmedディレクトリ
        output_path: 出力先のmetadata.jsonlパス
        valid_ratio: 検証セットの割合
        max_speakers: 最大話者数
        max_utts_per_speaker: 話者あたりの最大発話数

    Returns:
        (train_count, valid_count): 生成したエントリ数
    """
    print(f"\n=== Scanning VCTK: {data_dir} ===")

    # 可能なパスをチェック
    possible_paths = [
        data_dir / "wav48_silence_trimmed",
        data_dir / "VCTK-Corpus-0.92" / "wav48_silence_trimmed",
        data_dir / "VCTK-Corpus" / "wav48_silence_trimmed",
        data_dir,  # 既にwav48_silence_trimmedの場合
    ]

    wav_dir = None
    for path in possible_paths:
        if path.exists() and any(path.iterdir()):
            wav_dir = path
            break

    if wav_dir is None:
        print(f"⚠ VCTK audio directory not found")
        return 0, 0

    print(f"Using audio directory: {wav_dir}")

    # 話者ディレクトリを取得
    speaker_dirs = sorted([d for d in wav_dir.iterdir() if d.is_dir()])

    if max_speakers:
        speaker_dirs = speaker_dirs[:max_speakers]

    print(f"Found {len(speaker_dirs)} speakers")

    # 全話者の音声ファイルを収集
    speaker_files = {}
    for speaker_dir in tqdm(speaker_dirs, desc="Collecting speaker files"):
        speaker_id = speaker_dir.name
        wav_files = sorted(speaker_dir.glob("*.wav"))

        if not wav_files:
            continue

        if max_utts_per_speaker:
            wav_files = wav_files[:max_utts_per_speaker]

        speaker_files[speaker_id] = wav_files

    if len(speaker_files) == 0:
        print(f"⚠ Warning: No speakers found.")
        return 0, 0

    train_entries = []
    valid_entries = []

    for speaker_id, wav_files in tqdm(speaker_files.items(), desc="Processing speakers"):
        # 検証セット用にランダムサンプリング
        num_valid = max(1, int(len(wav_files) * valid_ratio))
        valid_indices = set(random.sample(range(len(wav_files)), num_valid))

        for i, wav_file in enumerate(wav_files):
            # 参照音声は同一話者の別発話を使用（StreamVC論文の自己再構成学習）
            if len(wav_files) > 1:
                ref_idx = (i + 1) % len(wav_files)
                ref_file = wav_files[ref_idx]
            else:
                ref_file = wav_file

            # wav_dirからの相対パスを取得
            source_rel = wav_file.relative_to(wav_dir)
            ref_rel = ref_file.relative_to(wav_dir)

            entry = {
                "id": f"{speaker_id}_{wav_file.stem}",
                "split": "valid" if i in valid_indices else "train",
                "source": str(source_rel),
                "reference": str(ref_rel),
                "speaker_id": speaker_id,
            }

            if i in valid_indices:
                valid_entries.append(entry)
            else:
                train_entries.append(entry)

    # メタデータを保存
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for entry in train_entries + valid_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"✓ Generated metadata: {output_path}")
    print(f"  Train: {len(train_entries)} entries")
    print(f"  Valid: {len(valid_entries)} entries")

    return len(train_entries), len(valid_entries)


def scan_jvs(
    data_dir: Path,
    output_path: Path,
    valid_ratio: float = 0.05,
    max_speakers: int = None,
    max_utts_per_speaker: int = None,
) -> Tuple[int, int]:
    """
    JVSデータセットをスキャンしてメタデータを生成

    Args:
        data_dir: JVSのルートディレクトリ (data/jvs/)
        output_path: 出力先のmetadata.jsonlパス
        valid_ratio: 検証セットの割合
        max_speakers: 最大話者数
        max_utts_per_speaker: 話者あたりの最大発話数

    Returns:
        (train_count, valid_count): 生成したエントリ数
    """
    print(f"\n=== Scanning JVS: {data_dir} ===")

    jvs_dir = data_dir / "jvs_ver1"

    if not jvs_dir.exists():
        print(f"⚠ JVS directory not found: {jvs_dir}")
        print("  Please download JVS corpus manually")
        return 0, 0

    # 話者ディレクトリを取得
    speaker_dirs = sorted([d for d in jvs_dir.iterdir() if d.is_dir() and d.name.startswith("jvs")])

    if max_speakers:
        speaker_dirs = speaker_dirs[:max_speakers]

    print(f"Found {len(speaker_dirs)} speakers")

    # 全話者の音声ファイルを収集
    speaker_files = {}
    for speaker_dir in tqdm(speaker_dirs, desc="Collecting speaker files"):
        speaker_id = speaker_dir.name

        # parallel100を優先的に使用（全話者で共通の発話）
        parallel_dir = speaker_dir / "parallel100" / "wav24kHz16bit"
        if not parallel_dir.exists():
            # parallel100がない場合は全wav24kHz16bitファイルを使用
            wav_files = sorted(speaker_dir.rglob("wav24kHz16bit/*.wav"))
        else:
            wav_files = sorted(parallel_dir.glob("*.wav"))

        if not wav_files:
            continue

        if max_utts_per_speaker:
            wav_files = wav_files[:max_utts_per_speaker]

        speaker_files[speaker_id] = wav_files

    if len(speaker_files) == 0:
        print(f"⚠ Warning: No speakers found.")
        return 0, 0

    train_entries = []
    valid_entries = []

    for speaker_id, wav_files in tqdm(speaker_files.items(), desc="Processing speakers"):
        # 検証セット用にランダムサンプリング
        num_valid = max(1, int(len(wav_files) * valid_ratio))
        valid_indices = set(random.sample(range(len(wav_files)), num_valid))

        for i, wav_file in enumerate(wav_files):
            # 参照音声は同一話者の別発話を使用（StreamVC論文の自己再構成学習）
            if len(wav_files) > 1:
                ref_idx = (i + 1) % len(wav_files)
                ref_file = wav_files[ref_idx]
            else:
                ref_file = wav_file

            # jvs_ver1からの相対パスを取得
            source_rel = wav_file.relative_to(jvs_dir)
            ref_rel = ref_file.relative_to(jvs_dir)

            entry = {
                "id": f"{speaker_id}_{wav_file.stem}",
                "split": "valid" if i in valid_indices else "train",
                "source": str(source_rel),
                "reference": str(ref_rel),
                "speaker_id": speaker_id,
            }

            if i in valid_indices:
                valid_entries.append(entry)
            else:
                train_entries.append(entry)

    # メタデータを保存
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for entry in train_entries + valid_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"✓ Generated metadata: {output_path}")
    print(f"  Train: {len(train_entries)} entries")
    print(f"  Valid: {len(valid_entries)} entries")

    return len(train_entries), len(valid_entries)


def main():
    parser = argparse.ArgumentParser(description="Generate metadata for StreamVC datasets")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Base directory for datasets (default: data/)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["libritts", "vctk", "jvs", "all"],
        default=["all"],
        help="Which datasets to process (default: all)"
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.05,
        help="Ratio of validation set (default: 0.05)"
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Maximum number of speakers per dataset (for quick testing)"
    )
    parser.add_argument(
        "--max-utts-per-speaker",
        type=int,
        default=None,
        help="Maximum utterances per speaker (for quick testing)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/valid split (default: 42)"
    )

    args = parser.parse_args()

    # シード設定
    random.seed(args.seed)

    print("=" * 60)
    print("StreamVC Metadata Generation")
    print("=" * 60)

    process_all = "all" in args.datasets
    total_train = 0
    total_valid = 0

    # LibriTTS
    if process_all or "libritts" in args.datasets:
        libritts_dir = args.data_dir / "libritts" / "LibriTTS" / "dev-clean"
        # dev-cleanがない場合は他のサブセットをチェック
        if not libritts_dir.exists():
            for subset in ["train-clean-100", "train-other-500", "test-clean"]:
                alt_dir = args.data_dir / "libritts" / "LibriTTS" / subset
                if alt_dir.exists():
                    libritts_dir = alt_dir
                    break

        output_path = args.data_dir / "libritts" / "metadata.jsonl"
        train, valid = scan_libritts(
            libritts_dir,
            output_path,
            args.valid_ratio,
            args.max_speakers,
            args.max_utts_per_speaker,
        )
        total_train += train
        total_valid += valid

    # VCTK
    if process_all or "vctk" in args.datasets:
        vctk_dir = args.data_dir / "vctk"
        output_path = args.data_dir / "vctk" / "metadata.jsonl"
        train, valid = scan_vctk(
            vctk_dir,
            output_path,
            args.valid_ratio,
            args.max_speakers,
            args.max_utts_per_speaker,
        )
        total_train += train
        total_valid += valid

    # JVS
    if process_all or "jvs" in args.datasets:
        jvs_dir = args.data_dir / "jvs"
        output_path = args.data_dir / "jvs" / "metadata.jsonl"
        train, valid = scan_jvs(
            jvs_dir,
            output_path,
            args.valid_ratio,
            args.max_speakers,
            args.max_utts_per_speaker,
        )
        total_train += train
        total_valid += valid

    # サマリー
    print("\n" + "=" * 60)
    print("Metadata Generation Summary")
    print("=" * 60)
    print(f"Total train entries: {total_train}")
    print(f"Total valid entries: {total_valid}")
    print(f"Total: {total_train + total_valid}")
    print("\nNext steps:")
    print("1. Review generated metadata.jsonl files")
    print("2. Run preprocessing to generate feature caches:")
    print("   python scripts/preprocess.py")


if __name__ == "__main__":
    main()
