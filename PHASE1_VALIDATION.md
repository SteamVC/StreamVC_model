# フェーズ1: RVQ Collapse修正 - 最小限の変更

## 実装内容

### 1. Pre-RVQ正規化（スケール整合）
**ファイル**: `src/streamvc/modules/decoder.py:122-137`

```python
# 1x1 Convでスケール調整
x_for_rvq = self.pre_rvq_conv(x)  # (B, C, T)

# 手動正規化: mean=0, std=1
x_for_rvq = x_for_rvq - x_for_rvq.mean(dim=-1, keepdim=True)
x_for_rvq = x_for_rvq / (x_for_rvq.std(dim=-1, keepdim=True) + 1e-5)
```

**根拠**:
- 問題: input std=0.31 vs codebook std=1.0 → 単一コード独占
- 修正: 正規化でstd=1に揃える → 公平な競争

### 2. Progressive RVQ
**ファイル**: `src/streamvc/modules/rvq.py:19,33-35,44-47`

```yaml
progressive_steps: 2000  # 2kステップごとに1段追加
```

**スケジュール**:
- 0-2k steps: quantizer 1段のみ
- 2k-4k steps: 2段
- 4k-6k steps: 3段
- ...
- 14k+ steps: 全8段

**根拠**:
- 初期段階でコードブック学習を安定化
- 段階的に表現力を増やす

### 3. Commitment Cost調整
**設定**: `commitment_cost: 0.5` (元: 0.25)

**根拠**:
- β↑でエンコーダが単一コードに過剰コミットするのを防止
- 必要に応じて1.0でABテスト可能

### 4. 安定化設定
- **Grad clip**: 既に `max_norm=1.0` で実装済み (`trainer.py:174`)
- **GAN OFF**: `adversarial_weight: 0.0`, `feature_matching_weight: 0.0`

## 合格基準（1-2k step時点）

### 必須条件:
| メトリクス | 目標 | 現状（崩壊時） |
|---------|------|--------------|
| **Perplexity** | 単調増加 5→10以上 | ~1.0（固定） |
| **Pre-RVQ std** | ≈1.0 | 0.31 |
| **単一コード独占** | 解消（usage >5%） | <0.1%（code 808のみ） |
| **MS-STFT Loss** | 下降傾向 | 停滞 |

### モニタリングメトリクス:
- `train/rvq_perplexity_q0` (第1段のみ2k stepまで)
- `train/rvq_usage_q0`
- `train/pre_rvq_std` (≈1.0を維持)
- `train/loss_stft`, `train/loss_l1`
- `train/num_active_quantizers` (progressive確認)

## 検証プロトコル

### トレーニングコマンド:
```bash
uv run python scripts/train.py --config configs/mps_training_validation.yaml
```

### モニタリング:
```bash
tensorboard --logdir runs/streamvc_mps_rvq_phase1/logs --port 6006
```

### チェックポイント:
- **1k step**: 第1段のみ、warmup直後の状態確認
- **2k step**: 第2段追加直前、第1段が健全か確認
- **5k step**: 第3段まで追加、全体的な改善確認
- **10k step**: 第6段まで、最終判定

### 各チェックポイントでの確認事項:

#### 1k step:
1. TensorBoardで確認:
   - `rvq_perplexity_q0` が **5以上**（理想: >10）
   - `rvq_usage_q0` が **>5%**
   - `pre_rvq_std` が **0.9-1.1**
   - `loss_stft` が下降中

2. Inference test:
   ```bash
   uv run python scripts/infer.py \
       --checkpoint runs/streamvc_mps_rvq_phase1/checkpoints/step_1000.pt \
       --config configs/mps_training_validation.yaml \
       --source <SOURCE_WAV> \
       --target-ref <TARGET_REF> \
       --output outputs/phase1/step_1000.wav \
       --device cpu
   ```
   - 音声が無音でないこと（std > 0.01）

#### 2k step:
- 第1段が安定（perplexity維持またはさらに増加）
- 第2段追加時にメトリクスが大きく乱れないこと

#### 5k/10k step:
- 複数段のperplexityが全て健全（各段 >5）
- MS-STFT/L1が継続的に改善
- 音声品質が向上

## 判定基準

### ✅ 成功（フェーズ2へ）:
- 1k stepで perplexity >5
- 2k stepで第1段が安定維持
- 5k stepで複数段が健全
- 音声が多様（無音でない）

### ⚠️ 部分成功（調整必要）:
- Perplexity改善はあるが<5 → commitment cost 1.0へ
- 特定段のみ崩壊 → progressive間隔調整

### ❌ 失敗（フェーズ2へ移行）:
- Perplexity <2で固定
- 依然として単一コード独占
- → より強力な手法が必要:
  - Cosine distance VQ
  - EMA codebook
  - Dead code reinitialization

## ABテスト準備

Commitment cost のABテストが必要な場合:

**設定A**: `commitment_cost: 0.5` (現在の設定)
**設定B**: `commitment_cost: 1.0`

```yaml
# configs/mps_training_validation_beta1.yaml
rvq:
  commitment_cost: 1.0
  progressive_steps: 2000
```

2並列で回して1k/2k stepでperplexityを比較。

## 変更箇所サマリ

| ファイル | 変更内容 | 理由 |
|---------|---------|------|
| `decoder.py:99,122-137` | Pre-RVQ正規化追加 | スケール整合 |
| `rvq.py:19,31,33-35,44-47` | Progressive RVQ | 段階的学習安定化 |
| `trainer.py:162-169,183-184` | Progressive制御 | RVQ段数管理 |
| `mps_training.yaml` | β=0.5, GAN OFF, progressive設定 | 安定化 |
| `mps_training_validation.yaml` | 検証用設定 | 早期確認 |

## 次ステップ

1. ✅ フェーズ1実装完了
2. ⏭️ 検証トレーニング開始（承認後）
3. ⏭️ 1k/2k/5k/10k stepで判定
4. ⏭️ 成功ならフェーズ2（GAN追加、長期学習）へ

---

**数学的保証**:

正規化により: `||normalize(x) - e||²` where `std(normalize(x)) = std(e) = 1.0`

→ スケール一致 → ユークリッド距離で公平な最近傍探索 → 多様なコード使用
