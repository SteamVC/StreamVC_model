# StreamVC セットアップガイド

このガイドでは、StreamVCの学習環境を構築する手順を説明します。

## 前提条件

- Python 3.9以上
- 十分なディスク容量（最低50GB、推奨100GB以上）
- （推奨）CUDA対応GPU

## 1. 環境構築

### 1.1 リポジトリのクローン

```bash
cd /Users/akatuki/streamVC
```

### 1.2 Pythonパッケージのインストール

```bash
# 仮想環境の作成（推奨）
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows

# 依存パッケージのインストール
pip install -e .
```

## 2. データセットの準備

### 2.1 HuBERTモデルのダウンロード

```bash
python scripts/download_datasets.py --datasets hubert
```

**ダウンロード内容:**
- HuBERT-BASE モデル（1.14GB）
- K-means量子化モデル（301KB）
- 保存先: `data/cache/hubert/`

### 2.2 音声データセットのダウンロード

#### LibriTTS（推奨・小規模テスト用）

```bash
# dev-cleanサブセット（1.2GB、40話者）
python scripts/download_datasets.py --datasets libritts --libritts-subset dev-clean
```

**その他のサブセット:**
```bash
# train-clean-100（24GB、247話者）
python scripts/download_datasets.py --datasets libritts --libritts-subset train-clean-100

# train-other-500（41.5GB、1,151話者）- 本格学習用
python scripts/download_datasets.py --datasets libritts --libritts-subset train-other-500
```

#### VCTK（オプション）

```bash
python scripts/download_datasets.py --datasets vctk
```

**ダウンロード内容:**
- VCTK-Corpus-0.92（516MB zip → 約10GB解凍後）
- 110話者、英語
- **注意:** 解凍に10-20分かかります

#### JVS（日本語・手動ダウンロード）

JVSコーパスは手動ダウンロードが必要です:

1. [JVS公式サイト](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)からダウンロード
2. `data/jvs/`に解凍
3. ディレクトリ構成: `data/jvs/jvs_ver1/jvs001/parallel100/wav24kHz16bit/...`

## 3. データの解凍確認

ダウンロードが完了したら、以下のコマンドで確認します:

```bash
# LibriTTSの確認
ls -lh data/libritts/LibriTTS/dev-clean/
# 出力例: 40個の話者ディレクトリ（19、26、...）

# VCTKの確認（ダウンロードした場合）
ls -lh data/vctk/VCTK-Corpus-0.92/wav48_silence_trimmed/
# 出力例: 110個の話者ディレクトリ（p225、p226、...）
```

### 3.1 手動解凍（VCTKが遅い場合）

ダウンロードスクリプトが遅い場合、手動で解凍できます:

```bash
# VCTKの手動解凍
cd data/vctk
unzip -q VCTK-Corpus-0.92.zip
cd ../..
```

## 4. メタデータの生成

音声ファイルをスキャンしてメタデータを生成します:

```bash
# LibriTTSのみ
python scripts/generate_metadata.py --datasets libritts

# 全データセット
python scripts/generate_metadata.py --datasets all

# 特定のデータセット
python scripts/generate_metadata.py --datasets libritts vctk
```

**生成されるファイル:**
- `data/libritts/metadata.jsonl`
- `data/vctk/metadata.jsonl`
- `data/jvs/metadata.jsonl`

**メタデータの確認:**
```bash
# LibriTTSのエントリ数確認
wc -l data/libritts/metadata.jsonl
# 出力例: 1447 data/libritts/metadata.jsonl

# 最初の1エントリを表示
head -1 data/libritts/metadata.jsonl | python -m json.tool
```

## 5. 特徴キャッシュの生成

HuBERTラベルとメタデータから特徴キャッシュを事前計算します:

```bash
# LibriTTSのtrainセット
python scripts/preprocess.py --dataset libri_tts --split train --device cpu

# validセット
python scripts/preprocess.py --dataset libri_tts --split valid --device cpu

# GPU使用の場合
python scripts/preprocess.py --dataset libri_tts --split train --device cuda
```

**生成されるファイル:**
- `data/cache/libri_tts/train/*.pt`（各サンプル1ファイル）
- `data/cache/libri_tts/valid/*.pt`

**処理時間の目安:**
- LibriTTS dev-clean（1447サンプル）: 約2-3分（CPU）
- VCTK全体（約45,000サンプル）: 約30-60分（CPU）

**キャッシュの確認:**
```bash
# 生成されたファイル数を確認
find data/cache/libri_tts/ -name "*.pt" | wc -l
# 出力例: 1447（train: 1380, valid: 67）

# 1つのキャッシュファイルを確認
python -c "
import torch
data = torch.load('data/cache/libri_tts/train/19_19_000000.pt')
print('Keys:', data.keys())
print('Shapes:', {k: v.shape for k, v in data.items()})
"
```

## 6. 設定ファイルの準備

学習設定を編集します:

```bash
# テスト用設定（100ステップ、小バッチサイズ）
cp configs/test_libritts.yaml configs/my_test.yaml

# 本格学習用設定
cp configs/default.yaml configs/my_config.yaml
```

**設定のカスタマイズ例:**

```yaml
# configs/my_config.yaml
data:
  datasets:
    - name: libri_tts
      root: data/libritts
      metadata: data/libritts/metadata.jsonl
      weight: 1.0
    # VCTKを追加する場合（コメント解除）
    # - name: vctk
    #   root: data/vctk
    #   metadata: data/vctk/metadata.jsonl
    #   weight: 0.5

training:
  batch_size: 16  # GPUメモリに応じて調整
  num_steps: 400000  # 学習ステップ数
  output_dir: runs/my_experiment
```

## 7. 学習の実行

### 7.1 テスト実行（100ステップ）

```bash
python scripts/train.py --config configs/test_libritts.yaml --device cpu
```

**期待される出力:**
```
Step 10/100 | Loss: 12.345 | Content: 4.56 | Recon: 7.89
Step 20/100 | Loss: 11.234 | Content: 4.23 | Recon: 7.01
...
Training complete!
```

### 7.2 本格学習

```bash
# CPUで学習（遅い）
python scripts/train.py --config configs/my_config.yaml --device cpu

# GPUで学習（推奨）
python scripts/train.py --config configs/my_config.yaml --device cuda
```

### 7.3 バックグラウンド実行

```bash
# nohupで実行
nohup python scripts/train.py --config configs/my_config.yaml --device cuda > train.log 2>&1 &

# ログを確認
tail -f train.log
```

### 7.4 学習の監視（TensorBoard）

```bash
# 別ターミナルで実行
tensorboard --logdir runs/streamvc_test/logs
       --port 6006 --host 0.0.0.0 2>&1 &
      echo "TensorBoard started on http://localhost:6006"

# ブラウザで開く
open http://localhost:6006
```

## 8. トラブルシューティング

### 8.1 データセットが見つからない

**エラー:** `FileNotFoundError: data/libritts/metadata.jsonl`

**解決策:**
```bash
# メタデータを再生成
python scripts/generate_metadata.py --datasets libritts
```

### 8.2 キャッシュファイルが見つからない

**エラー:** `RuntimeError: No cache files found in data/cache/libri_tts/train`

**解決策:**
```bash
# 特徴キャッシュを再生成
python scripts/preprocess.py --dataset libri_tts --split train --device cpu
```

### 8.3 メモリ不足

**エラー:** `RuntimeError: CUDA out of memory`

**解決策:**
```yaml
# configs/my_config.yamlでバッチサイズを削減
training:
  batch_size: 4  # 16 → 4に変更
```

### 8.4 HuBERTモデルが見つからない

**エラー:** `FileNotFoundError: data/cache/hubert/hubert_base_ls960.pt`

**解決策:**
```bash
python scripts/download_datasets.py --datasets hubert
```

### 8.5 VCTKの解凍が遅い

**現象:** `python scripts/download_datasets.py --datasets vctk`が1時間以上実行される

**解決策:**
```bash
# プロセスをキル
pkill -f "download_datasets.py --datasets vctk"

# 手動解凍
cd data/vctk
unzip -q VCTK-Corpus-0.92.zip
cd ../..

# メタデータ生成
python scripts/generate_metadata.py --datasets vctk
```

## 9. ディレクトリ構造（完成形）

```
/Users/akatuki/streamVC/
├── data/
│   ├── cache/
│   │   ├── hubert/
│   │   │   ├── hubert_base_ls960.pt    # 1.14GB
│   │   │   └── km100.bin               # 301KB
│   │   ├── libri_tts/
│   │   │   ├── train/                  # 1380 .pt files
│   │   │   └── valid/                  # 67 .pt files
│   │   └── vctk/
│   │       ├── train/
│   │       └── valid/
│   ├── libritts/
│   │   ├── metadata.jsonl              # 1447 entries
│   │   └── LibriTTS/
│   │       └── dev-clean/              # 40 speakers
│   ├── vctk/
│   │   ├── metadata.jsonl
│   │   └── VCTK-Corpus-0.92/
│   │       └── wav48_silence_trimmed/  # 110 speakers
│   └── jvs/
│       ├── metadata.jsonl
│       └── jvs_ver1/                   # 100 speakers
├── runs/
│   └── my_experiment/
│       ├── checkpoints/                # 学習済みモデル
│       └── logs/                       # TensorBoardログ
└── configs/
    ├── default.yaml                    # 本格学習用
    └── test_libritts.yaml              # テスト用
```

## 10. 次のステップ

学習が完了したら:

1. **チェックポイントの確認:**
   ```bash
   ls -lh runs/my_experiment/checkpoints/
   ```

2. **推論の実行:**
   ```bash
   # 推論スクリプトを作成する必要があります
   python scripts/inference.py --checkpoint runs/my_experiment/checkpoints/step_100000.pt
   ```

3. **音声品質の評価:**
   - MCD (Mel-Cepstral Distortion)
   - MOS (Mean Opinion Score)
   - リアルタイム性能の測定

## 参考情報

- **アーキテクチャ仕様:** `docs/streamvc_architecture.md`
- **論文:** [StreamVC: Real-Time Low-Latency Voice Conversion](https://arxiv.org/abs/...)
- **問題報告:** GitHub Issues

---

**作成日:** 2025-11-05
**対象バージョン:** StreamVC v0.1.0
