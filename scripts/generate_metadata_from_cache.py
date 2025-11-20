#!/usr/bin/env python3
"""
キャッシュファイルからmetadata.jsonlを生成するスクリプト

キャッシュファイル名から元の音声ファイル構造を復元し、
同一話者ペアリングでmetadata.jsonlを生成します。
"""

import json
import random
from pathlib import Path
from collections import defaultdict


def main():
    cache_dir = Path("data/cache/libri_tts")
    output_path = Path("data/libritts/metadata.jsonl")

    # train/validのキャッシュファイルを収集
    train_files = list((cache_dir / "train").glob("*.pt"))
    valid_files = list((cache_dir / "valid").glob("*.pt"))

    print(f"Found {len(train_files)} train cache files")
    print(f"Found {len(valid_files)} valid cache files")

    # 話者ごとにファイルをグループ化
    def group_by_speaker(files):
        speaker_files = defaultdict(list)
        for f in files:
            # ファイル名: {speaker}_{speaker}_{chapter}_{utt}_{seg}.pt
            parts = f.stem.split("_")
            speaker_id = parts[0]
            # 元のパスを復元: {speaker}/{chapter}/{speaker}_{chapter}_{utt}_{seg}.wav
            chapter_id = parts[2]
            utt_id = parts[3]
            seg_id = parts[4]
            original_name = f"{speaker_id}_{chapter_id}_{utt_id}_{seg_id}"
            rel_path = f"{speaker_id}/{chapter_id}/{original_name}.wav"
            speaker_files[speaker_id].append(rel_path)
        return speaker_files

    train_by_speaker = group_by_speaker(train_files)
    valid_by_speaker = group_by_speaker(valid_files)

    entries = []

    # Train entries
    for speaker_id, files in train_by_speaker.items():
        files = sorted(files)
        for i, source in enumerate(files):
            # 同一話者の別発話をリファレンスに使用
            if len(files) > 1:
                ref_idx = (i + 1) % len(files)
                reference = files[ref_idx]
            else:
                reference = source

            entry = {
                "id": f"{speaker_id}_{Path(source).stem}",
                "split": "train",
                "source": source,
                "reference": reference,
                "speaker_id": speaker_id,
            }
            entries.append(entry)

    # Valid entries
    for speaker_id, files in valid_by_speaker.items():
        files = sorted(files)
        for i, source in enumerate(files):
            if len(files) > 1:
                ref_idx = (i + 1) % len(files)
                reference = files[ref_idx]
            else:
                reference = source

            entry = {
                "id": f"{speaker_id}_{Path(source).stem}",
                "split": "valid",
                "source": source,
                "reference": reference,
                "speaker_id": speaker_id,
            }
            entries.append(entry)

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    train_count = sum(1 for e in entries if e["split"] == "train")
    valid_count = sum(1 for e in entries if e["split"] == "valid")

    print(f"\nGenerated metadata: {output_path}")
    print(f"  Train: {train_count} entries")
    print(f"  Valid: {valid_count} entries")


if __name__ == "__main__":
    main()
