# StreamVC: Real-time Any-to-Many Voice Conversion

リアルタイム多話者音声変換システム

## 特徴

- リアルタイムストリーミング対応
- Any-to-Many Voice Conversion
- HuBERT特徴量ベース
- RVQ (Residual Vector Quantization) による高品質合成
- Google Colab対応

## クイックスタート

### ローカル環境でのセットアップ

```bash
# 1. リポジトリのクローン
git clone https://github.com/YOUR_USERNAME/streamVC.git
cd streamVC

# 2. 依存パッケージのインストール
pip install -e .

# 3. データセットのダウンロード
python scripts/download_datasets.py --datasets hubert libritts

# 4. メタデータの生成
python scripts/generate_metadata.py --datasets libritts

# 5. 前処理（特徴キャッシュの生成）
python scripts/preprocess.py --dataset libri_tts --split train
python scripts/preprocess.py --dataset libri_tts --split valid

# 6. 学習の実行
python scripts/train.py --config configs/default.yaml
```

### Google Colabでの学習

Google Colab上で学習を実行する場合は、以下の手順に従ってください。

#### 1. ローカルでデータセットを準備

```bash
# rcloneのセットアップ
./scripts/setup_rclone.sh

# データセットのダウンロードと前処理
python scripts/download_datasets.py --datasets hubert libritts
python scripts/generate_metadata.py --datasets libritts
python scripts/preprocess.py --dataset libri_tts --split train
python scripts/preprocess.py --dataset libri_tts --split valid

# Google Driveにアップロード
./scripts/upload_dataset_to_gdrive.sh
```

#### 2. GitHubにリポジトリをpush

```bash
git add .
git commit -m "Add training setup"
git push origin main
```

#### 3. Colabで学習

1. `colab_setup.ipynb`をGoogle Colabで開く
2. ノートブックの指示に従って実行
3. Checkpointは自動的にGoogle Driveにバックアップされます

## ディレクトリ構造

```
streamVC/
├── src/streamvc/          # ソースコード
│   ├── pipeline.py        # メインパイプライン
│   ├── trainer.py         # 学習ループ
│   ├── modules/           # モデルコンポーネント
│   └── data/              # データローダー
├── scripts/               # 実行スクリプト
│   ├── download_datasets.py
│   ├── generate_metadata.py
│   ├── preprocess.py
│   ├── train.py
│   ├── setup_rclone.sh
│   └── upload_dataset_to_gdrive.sh
├── configs/               # 設定ファイル
│   ├── default.yaml
│   └── colab_gpu_training.yaml
├── data/                  # データセット（.gitignore）
│   ├── cache/hubert/
│   ├── libritts/
│   └── vctk/
├── runs/                  # 学習結果（.gitignore）
└── colab_setup.ipynb      # Colab用ノートブック
```

## Google Drive構成

```
MyDrive/
└── streamVC/
    ├── data/              # データセット
    │   ├── cache/hubert/
    │   ├── libritts/
    │   └── vctk/
    └── checkpoints/       # 学習checkpoint（自動バックアップ）
```

## 設定ファイル

学習設定は`configs/`ディレクトリのYAMLファイルで管理します。

### 主な設定項目

```yaml
# configs/default.yaml
data:
  datasets:
    - name: libri_tts
      root: data/libritts
      metadata: data/libritts/metadata.jsonl
      weight: 1.0

training:
  batch_size: 16
  num_steps: 400000
  ckpt_interval: 10000
  eval_interval: 1000
  output_dir: runs/streamvc

model:
  num_hubert_labels: 100
  decoder:
    rvq:
      num_quantizers: 8
```

## データセット

### サポートしているデータセット

- **LibriTTS**: 英語、多話者TTS用コーパス
- **VCTK**: 英語、110話者
- **JVS**: 日本語、100話者（手動ダウンロード）

### データセットのダウンロード

```bash
# LibriTTS（テスト用）
python scripts/download_datasets.py --datasets libritts --libritts-subset dev-clean

# LibriTTS（本格学習用）
python scripts/download_datasets.py --datasets libritts --libritts-subset train-other-500

# VCTK
python scripts/download_datasets.py --datasets vctk

# 全て
python scripts/download_datasets.py --datasets all
```

## 学習

### ローカルでの学習

```bash
# GPU（CUDA）
python scripts/train.py --config configs/default.yaml --device cuda

# Apple Silicon（MPS）
python scripts/train.py --config configs/mps_training.yaml --device mps

# CPU
python scripts/train.py --config configs/cpu_training.yaml --device cpu
```

### Colabでの学習（Google Driveバックアップ付き）

```bash
python scripts/train.py \
    --config configs/colab_gpu_training.yaml \
    --device cuda \
    --gdrive-backup /content/drive/MyDrive/streamVC/checkpoints
```

Checkpointは自動的にGoogle Driveにバックアップされるため、Colabのセッションが切れても安全です。

### TensorBoard

```bash
tensorboard --logdir runs/streamvc/logs --port 6006
```

ブラウザで http://localhost:6006 を開く

## Checkpointの管理

### Checkpointの保存

- デフォルトでは `ckpt_interval` ごとに保存（例: 10,000ステップごと）
- 保存先: `runs/[experiment_name]/checkpoints/step_*.pt`
- Google Driveバックアップが有効な場合、自動的にコピーされます

### Checkpointからの再開

```python
# train.pyで自動再開機能を実装する場合
python scripts/train.py --resume runs/streamvc/checkpoints/step_100000.pt
```

## 推論

```bash
# 推論スクリプト（準備中）
python scripts/infer.py \
    --checkpoint runs/streamvc/checkpoints/step_100000.pt \
    --source input.wav \
    --target-speaker 123 \
    --output output.wav
```

## トラブルシューティング

### Q: データセットが見つからない
```bash
# メタデータを再生成
python scripts/generate_metadata.py --datasets libritts
```

### Q: キャッシュファイルが見つからない
```bash
# 前処理を再実行
python scripts/preprocess.py --dataset libri_tts --split train
```

### Q: メモリ不足（OOM）
```yaml
# configs/your_config.yaml
training:
  batch_size: 8  # バッチサイズを削減
```

### Q: Colab接続が切れた
Colabノートブックの「11. 学習の再開」セルを実行してください。Checkpointから自動的に再開されます。

### Q: rcloneの認証エラー
```bash
# 再認証
rclone config reconnect gdrive:
```

## 開発

### テストの実行

```bash
# 簡単なテスト実行（100ステップ）
python scripts/train.py --config configs/test_libritts.yaml --device cpu
```

### コードフォーマット

```bash
pip install black isort
black src/ scripts/
isort src/ scripts/
```

## ライセンス

このプロジェクトのライセンスについては、LICENSEファイルを参照してください。

## 引用

```bibtex
@article{streamvc2025,
  title={StreamVC: Real-Time Low-Latency Voice Conversion},
  author={Your Name},
  year={2025}
}
```

## 参考資料

- [セットアップガイド詳細](SETUP.md)
- [アーキテクチャ仕様](docs/streamvc_architecture.md)
- [開発計画](PHASE_PLAN.md)

## 貢献

Issue、Pull Requestを歓迎します。

---

**作成日**: 2025-11-14
**バージョン**: 0.1.0
