# Phase 1-EMA-GAN バグ分析レポート

## 発見されたバグ一覧

### バグ1: Progressive RVQ - `num_active_quantizers`保存漏れ (重大)

**症状**:
- 学習時: Step 4000で3つのQuantizer有効
- 推論時: デフォルト値8が使用され、未学習のCodebook 2-7も使用
- 結果: 推論時にノイズが生成される

**原因**:
- `trainer.py:save_checkpoint()` - `num_active_quantizers`を保存していなかった
- `trainer.py:load_checkpoint()` - `num_active_quantizers`を復元していなかった
- `infer.py:load_checkpoint()` - `num_active_quantizers`を復元していなかった

**修正**:
✅ `trainer.py:429` - checkpointに`num_active_quantizers`を追加
✅ `trainer.py:454-455` - load時に`num_active_quantizers`を復元
✅ `infer.py:35-38` - load時に`num_active_quantizers`を復元

---

### バグ2: EMA更新の二重正規化 (致命的)

**症状**:
- Codebook 0: std=1.5078, 範囲[-72, +69] (正常: std≈1.0, 範囲≈[-4, +4])
- Codebook 0の使用パターン: max_usage=7864.8 (異常に偏っている)
- Dead codes: 867/1024 (84.7%)

**原因** (`rvq.py:160-167`):
```python
# EMA update
self.cluster_size[q_idx].add_(cluster_size_update, alpha=1 - decay)
self.embed_avg[q_idx].add_(embed_sum, alpha=1 - decay)  # embed_sum is SUM

# Update codebook
n = self.cluster_size[q_idx].unsqueeze(1).clamp(min=1.0)
updated_codebook = self.embed_avg[q_idx] / n  # ← 二重正規化
self.codebooks[q_idx].data.copy_(updated_codebook)
```

**問題点**:
1. `embed_sum`は**累積和**であり、平均ではない
2. `embed_avg`に累積和のEMAを保存している
3. これを`cluster_size`で割ると、**二重に正規化**してしまう

**数式で説明**:
```
正しいEMA更新:
  embed_avg[k] = decay * embed_avg[k] + (1-decay) * sum(embeddings for code k)
  codebook[k] = embed_avg[k] / cluster_size[k]

しかし、cluster_sizeもEMAで更新されている:
  cluster_size[k] = decay * cluster_size[k] + (1-decay) * count(code k)

つまり:
  codebook[k] = [decay * embed_avg[k] + (1-decay) * embed_sum[k]] / [decay * cluster_size[k] + (1-decay) * count[k]]
              ≠ embed_sum[k] / count[k]  (正しい平均)

結果:
  - 高頻度code (cluster_size大) → codebook値が過度に小さくなる
  - 低頻度code (cluster_size小) → codebook値が過度に大きくなる
  - 使用頻度の偏りが、スケールの異常につながる
```

**実測値がこれを裏付ける**:
- Q0: max_usage=7864.8 → 特定のcodeに集中
- Q0のstd=1.5078 (他のQ2-Q7はstd≈1.0) → 異常なスケール
- Dead codes 84.7% → 使用が極端に偏る

**正しい実装 (DAC論文)**:
```python
# Option A: cluster_sizeもEMAではなく、累積カウントを保持
self.cluster_size[q_idx] = cluster_size_update  # No EMA decay
self.embed_avg[q_idx].mul_(decay).add_(embed_sum, alpha=1 - decay)
updated_codebook = self.embed_avg[q_idx] / self.cluster_size[q_idx].unsqueeze(1).clamp(min=1.0)

# Option B: EMAを使うなら、割り算をしない
self.cluster_size[q_idx].mul_(decay).add_(cluster_size_update, alpha=1 - decay)
self.embed_avg[q_idx].mul_(decay).add_(embed_sum, alpha=1 - decay)
# cluster_sizeで割らずに、embed_avgをそのままcodebookとして使用
# (ただし、これは数学的に不正確)

# Option C (推奨): 移動平均を正しく計算
# cluster_sizeは累積カウント、embed_sumは累積和
# Laplace smoothingを使って安定化
n = self.cluster_size[q_idx].unsqueeze(1) + 1e-5
updated_codebook = self.embed_avg[q_idx] / n
```

**推奨修正** (DACの実装に準拠):
```python
def ema_update(self, q_idx: int, indices: torch.Tensor, flat_input: torch.Tensor) -> None:
    if not self.config.use_ema or not self.training:
        return

    decay = self.config.ema_decay

    with torch.no_grad():
        # Count usage per code
        cluster_size_update = torch.bincount(
            indices,
            minlength=self.config.codebook_size
        ).float()

        # Compute sum of embeddings per code
        embed_sum = torch.zeros_like(self.embed_avg[q_idx])
        indices_expanded = indices.unsqueeze(1).expand(-1, flat_input.shape[1])
        embed_sum.scatter_add_(0, indices_expanded, flat_input)

        # EMA update for both cluster_size and embed_avg
        self.cluster_size[q_idx].mul_(decay).add_(cluster_size_update, alpha=1 - decay)
        self.embed_avg[q_idx].mul_(decay).add_(embed_sum, alpha=1 - decay)

        # Update codebook: divide embed_avg by cluster_size with Laplace smoothing
        # この計算は数学的には不正確だが、実用的には機能する
        # 正しくは、embed_avgとcluster_sizeを別々にEMA追跡してから割るべき
        n = self.cluster_size[q_idx].unsqueeze(1).clamp(min=1.0)
        updated_codebook = self.embed_avg[q_idx] / n
        self.codebooks[q_idx].data.copy_(updated_codebook)
```

**しかし、この実装も不完全**: EMAされたcluster_sizeで、EMAされたembed_avgを割ることは、数学的には正しくない。

**完全に正しい実装**:
```python
# 各codeについて、平均ベクトルをEMAで追跡する
# つまり、embed_avgには「平均」を保存し、cluster_sizeは使わない

# EMA update (per-code mean tracking)
for code_idx in used_codes:
    mask = (indices == code_idx)
    code_embeddings = flat_input[mask]  # (N, C)
    code_mean = code_embeddings.mean(dim=0)  # (C,)

    # Update EMA of mean
    self.codebooks[q_idx][code_idx].mul_(decay).add_(code_mean, alpha=1 - decay)
```

しかし、これはループが必要で非効率。

**実用的な妥協案**: 現在の実装を維持しつつ、Laplace smoothing強化
```python
# Stronger Laplace smoothing to prevent extreme values
n = self.cluster_size[q_idx].unsqueeze(1).clamp(min=10.0)  # min=1.0 → min=10.0
updated_codebook = self.embed_avg[q_idx] / n
```

---

### バグ3: Progressive RVQ タイミング問題 (軽微)

**症状**:
- Step 4000のcheckpointでCodebook 2が完全未使用 (cluster_size=0)
- 期待: 3つのquantizer有効 (Q0, Q1, Q2)

**原因** (`trainer.py:389`):
```python
for batch in train_loader:
    # Update active quantizers based on training step
    if progressive_steps > 0:
        num_active = min(1 + self.step // progressive_steps, ...)
        self.pipeline.decoder.rvq.set_num_active_quantizers(num_active)

    # ... training ...

    self.step += 1  # ← ループの最後でstep increment
    if self.step % self.config.training.ckpt_interval == 0:
        self.save_checkpoint(...)  # ← step increment直後に保存
```

**タイムライン**:
```
Step 3998: num_active=2
Step 3999: num_active=2
Step 4000: num_active=3 ← Q2有効化
  → train_step実行 (1 iteration分のみ)
  → self.step += 1 (step=4001に)
  → checkpoint保存 (step=4001として保存されるが、ファイル名はstep_4000.pt)
```

**問題**:
- Checkpointのstepフィールドは4000だが、実際にはstep 4001の状態
- Q2は実質1 iteration分しか学習されていない → cluster_size=0

**影響**: 軽微 (数iterationの遅れ程度)

**修正は不要**: Progressive RVQの性質上、新しいquantizerは徐々に学習されるため、1 iterationの遅れは問題ない。

ただし、より正確なログのためには:
```python
# ckpt_intervalチェックをstep increment前に移動
if self.step % self.config.training.ckpt_interval == 0:
    self.save_checkpoint(ckpt_dir / f"step_{self.step}.pt")

self.step += 1
```

---

## 修正の優先度

### 優先度S (即座に修正が必要)
✅ **バグ1: num_active_quantizers保存漏れ** - 修正済み

### 優先度A (学習再開前に修正が必要)
❌ **バグ2: EMA更新の二重正規化** - 未修正
  - このバグにより、Codebook 0が異常なスケールになり、学習が不安定
  - Dead codesが大量発生し、codebook利用効率が低下

### 優先度C (任意)
- **バグ3: Progressive RVQタイミング** - 実害は小さい、修正不要

---

## 推奨アクション

1. **バグ2 (EMA二重正規化)の修正**を議論して決定
   - Option A: 正しい数学的実装に修正 (複雑)
   - Option B: Laplace smoothingを強化 (min=1.0 → min=10.0) (簡単)
   - Option C: EMAを無効化して gradient-based VQ に戻す (use_ema=False)

2. 修正後、**学習を最初からやり直す**
   - Step 4000のcheckpointは、Codebook 0が異常なため使用不可

3. 学習中のモニタリング強化
   - Codebookの統計 (mean, std, min, max) をログ
   - Dead codes数をログ
   - 使用頻度の偏り (max_usage / mean_usage) をログ

---

## バグ2の修正案の詳細

### Option A: 正しい数学的実装 (推奨)

```python
def ema_update(self, q_idx: int, indices: torch.Tensor, flat_input: torch.Tensor) -> None:
    if not self.config.use_ema or not self.training:
        return

    decay = self.config.ema_decay

    with torch.no_grad():
        # Compute per-code statistics for current batch
        cluster_size_batch = torch.bincount(indices, minlength=self.config.codebook_size).float()

        embed_sum_batch = torch.zeros_like(self.embed_avg[q_idx])
        indices_expanded = indices.unsqueeze(1).expand(-1, flat_input.shape[1])
        embed_sum_batch.scatter_add_(0, indices_expanded, flat_input)

        # Compute batch mean per code (avoiding division by zero)
        mask = cluster_size_batch > 0
        embed_mean_batch = torch.zeros_like(self.embed_avg[q_idx])
        embed_mean_batch[mask] = embed_sum_batch[mask] / cluster_size_batch[mask].unsqueeze(1)

        # EMA update of codebook (tracking mean, not sum)
        # For codes used in this batch: update with EMA
        # For codes not used: keep existing value (no decay)
        self.codebooks[q_idx].data[mask] = (
            decay * self.codebooks[q_idx].data[mask] +
            (1 - decay) * embed_mean_batch[mask]
        )

        # Update cluster_size for monitoring (optional)
        self.cluster_size[q_idx].mul_(decay).add_(cluster_size_batch, alpha=1 - decay)
```

**利点**:
- 数学的に正しい
- スケールが安定する
- Dead codesが発生しにくい

**欠点**:
- maskベースの操作が必要

### Option B: Laplace smoothing強化 (簡易)

```python
# rvq.py:165 を修正
n = self.cluster_size[q_idx].unsqueeze(1).clamp(min=10.0)  # 1.0 → 10.0
```

**利点**:
- 修正が簡単
- ある程度スケールを安定化

**欠点**:
- 根本的な問題は解決しない
- Dead codesは依然として発生

### Option C: EMA無効化 (回避策)

```yaml
# config
rvq:
  use_ema: false
```

**利点**:
- バグを回避できる
- Gradient-based VQは理論的に正しい

**欠点**:
- EMAの利点(安定性、高速収束)を失う
- Commitment lossの調整が必要
