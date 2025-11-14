#!/bin/bash
# ローカルのデータセットをGoogle Driveにアップロードするスクリプト

set -e

REMOTE_NAME="akatuki"
REMOTE_BASE="streamVC"
LOCAL_DATA_DIR="data"

echo "=== Upload Dataset to Google Drive ==="
echo ""
echo "This script will upload the following to Google Drive:"
echo "  Local:  ${LOCAL_DATA_DIR}/"
echo "  Remote: ${REMOTE_NAME}:${REMOTE_BASE}/data/"
echo ""

# rcloneが設定されているか確認
if ! rclone listremotes | grep -q "^${REMOTE_NAME}:"; then
    echo "Error: rclone remote '${REMOTE_NAME}' is not configured"
    echo "Please run: ./scripts/setup_rclone.sh"
    exit 1
fi

# データディレクトリの存在確認
if [ ! -d "${LOCAL_DATA_DIR}" ]; then
    echo "Error: ${LOCAL_DATA_DIR}/ directory not found"
    echo "Please download datasets first: python scripts/download_datasets.py"
    exit 1
fi

# アップロードするファイルのサイズを確認
echo "Calculating upload size..."
du -sh ${LOCAL_DATA_DIR}

echo ""
read -p "Continue with upload? (y/N): " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Upload cancelled"
    exit 0
fi

echo ""
echo "=== Uploading... ==="
echo "Note: This may take several hours depending on dataset size"
echo ""

# HuBERTモデルをアップロード
if [ -d "${LOCAL_DATA_DIR}/cache/hubert" ]; then
    echo "Uploading HuBERT models..."
    rclone copy ${LOCAL_DATA_DIR}/cache/hubert \
        ${REMOTE_NAME}:${REMOTE_BASE}/data/cache/hubert \
        --progress \
        --transfers 4 \
        --checkers 8 \
        --exclude "*.tar.gz" \
        --exclude "*.zip"
    echo "✓ HuBERT models uploaded"
fi

# LibriTTSをアップロード
if [ -d "${LOCAL_DATA_DIR}/libritts" ]; then
    echo ""
    echo "Uploading LibriTTS..."
    rclone copy ${LOCAL_DATA_DIR}/libritts \
        ${REMOTE_NAME}:${REMOTE_BASE}/data/libritts \
        --progress \
        --transfers 4 \
        --checkers 8 \
        --exclude "*.tar.gz" \
        --exclude "*.zip"
    echo "✓ LibriTTS uploaded"
fi

# VCTKをアップロード（存在する場合）
if [ -d "${LOCAL_DATA_DIR}/vctk" ]; then
    echo ""
    echo "Uploading VCTK..."
    rclone copy ${LOCAL_DATA_DIR}/vctk \
        ${REMOTE_NAME}:${REMOTE_BASE}/data/vctk \
        --progress \
        --transfers 4 \
        --checkers 8 \
        --exclude "*.tar.gz" \
        --exclude "*.zip"
    echo "✓ VCTK uploaded"
fi

# JVSをアップロード（存在する場合）
if [ -d "${LOCAL_DATA_DIR}/jvs" ]; then
    echo ""
    echo "Uploading JVS..."
    rclone copy ${LOCAL_DATA_DIR}/jvs \
        ${REMOTE_NAME}:${REMOTE_BASE}/data/jvs \
        --progress \
        --transfers 4 \
        --checkers 8 \
        --exclude "*.tar.gz" \
        --exclude "*.zip"
    echo "✓ JVS uploaded"
fi

# 前処理済みキャッシュをアップロード（存在する場合）
if [ -d "${LOCAL_DATA_DIR}/cache/libri_tts" ]; then
    echo ""
    echo "Uploading preprocessed LibriTTS cache..."
    rclone copy ${LOCAL_DATA_DIR}/cache/libri_tts \
        ${REMOTE_NAME}:${REMOTE_BASE}/data/cache/libri_tts \
        --progress \
        --transfers 8 \
        --checkers 16
    echo "✓ LibriTTS cache uploaded"
fi

if [ -d "${LOCAL_DATA_DIR}/cache/vctk" ]; then
    echo ""
    echo "Uploading preprocessed VCTK cache..."
    rclone copy ${LOCAL_DATA_DIR}/cache/vctk \
        ${REMOTE_NAME}:${REMOTE_BASE}/data/cache/vctk \
        --progress \
        --transfers 8 \
        --checkers 16
    echo "✓ VCTK cache uploaded"
fi

echo ""
echo "=== Upload Complete! ==="
echo ""
echo "Verify upload:"
echo "  rclone ls ${REMOTE_NAME}:${REMOTE_BASE}/data/ --max-depth 2"
echo ""
echo "Google Drive structure:"
echo "  ${REMOTE_BASE}/"
echo "  └── data/"
echo "      ├── cache/"
echo "      │   ├── hubert/"
echo "      │   ├── libri_tts/"
echo "      │   └── vctk/"
echo "      ├── libritts/"
echo "      ├── vctk/"
echo "      └── jvs/"
echo ""
echo "Next: Use the Colab notebook to start training"
