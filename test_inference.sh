#!/bin/bash

# テスト用の音声ファイルを選択
SOURCE_SPEAKER="1069"
TARGET_SPEAKER="103"

SOURCE_WAV=$(find data/libritts/LibriTTS/train-clean-100/${SOURCE_SPEAKER} -name "*.wav" | head -1)
TARGET_REF=$(find data/libritts/LibriTTS/train-clean-100/${TARGET_SPEAKER} -name "*.wav" | head -1)

echo "Source: $SOURCE_WAV"
echo "Target Ref: $TARGET_REF"

# 出力ディレクトリ
OUTPUT_DIR="outputs/inference_test"
mkdir -p $OUTPUT_DIR

# チェックポイントリスト
CHECKPOINTS=(
    "runs/streamvc_mps_clean100_fixed/checkpoints/step_5000.pt"
    "runs/streamvc_mps_clean100_fixed/checkpoints/step_10000.pt"
    "runs/streamvc_mps_clean100_balanced/checkpoints/step_5000.pt"
)

# 各チェックポイントで推論
for CKPT in "${CHECKPOINTS[@]}"; do
    if [ -f "$CKPT" ]; then
        CKPT_NAME=$(basename $(dirname $(dirname $CKPT)))_$(basename $CKPT .pt)
        echo ""
        echo "=========================================="
        echo "Testing: $CKPT_NAME"
        echo "=========================================="
        
        # 設定ファイルを判定
        if [[ "$CKPT" == *"balanced"* ]]; then
            CONFIG="configs/mps_training_balanced.yaml"
        else
            CONFIG="configs/mps_training_fixed.yaml"
        fi
        
        uv run python scripts/infer.py \
            --checkpoint "$CKPT" \
            --config "$CONFIG" \
            --source "$SOURCE_WAV" \
            --target-ref "$TARGET_REF" \
            --output "$OUTPUT_DIR/${CKPT_NAME}.wav" \
            --device mps
    else
        echo "Checkpoint not found: $CKPT"
    fi
done

echo ""
echo "=========================================="
echo "All inference tests completed!"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="

