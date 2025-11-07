# Phase 1詳細分析と改善提案

**日付**: 2025-11-07
**目的**: Phase 1の問題点を詳細分析し、調査済みベストプラクティスに基づく改善策を提案

---

## TL;DR

### Phase 1の問題点

1. **Perplexity継続的低下**: 34.59 → 4.92 (-86%)、単調減少
2. **Q0のみ学習**: Q1-Q7のPerplexityデータなし（未使用？）
3. **RVQ Loss増加**: 後半で2.3-2.5に増加（量子化誤差増大）
4. **しかしCodebookは健全**: ノルム分布正常、Dead codeなし

### 根本原因の仮説

❌ **Codebook崩壊ではない** → ノルム分布が健全
✅ **少数のCodeのみ使用** → Perplexity低下 = 使用Codeが偏る
✅ **Progressive RVQが機能していない** → Q1-Q7が使われていない可能性

### 推奨する改善策

🎯 **Option 1: EMA更新 + Code balancing** (推奨度★★★★★)
🎯 **Option 2: Progressive RVQの修正** (推奨度★★★★☆)
🎯 **Option 3: Commitment loss調整** (推奨度★★★☆☆)

---

## 1. Phase 1 @ 6K 詳細分析

### 1.1 Perplexity推移の詳細

```
区間        Mean    Min-Max      Std     傾向
0-1K:      19.00   8.59-34.59   8.80    初期高、急降下
1K-2K:     13.53   9.65-19.46   2.82    継続低下
2K-3K:      8.56   6.81-12.70   1.73    さらに低下
3K-4K:      7.09   6.77-9.29    0.74    安定化？
4K-5K:      6.13   4.76-8.32    1.15    低下継続
5K-6K:      5.40   4.79-6.16    0.48    最低、安定
```

**観察**:
- **単調減少**: ピーク(step 0, 34.59) → 最終(step 6100, 4.92)
- **収束傾向なし**: 5K-6Kでもわずかに低下継続
- **標準偏差も減少**: 初期8.80 → 最終0.48（使用パターンが固定化）

**解釈**:
- Codebookの**一部のみが使用される**状態に収束
- **Winner-takes-all**現象（最近傍の数個のCodeのみ選ばれる）
- さらに学習を続けても改善しない可能性

### 1.2 RVQ Loss推移

```
区間        RVQ Loss   変化
0-1K:      1.1346     -
1K-2K:     1.0830     -4.5%
2K-3K:     1.6068     +48.3%  ← 急増
3K-4K:     1.1662     -27.4%
4K-5K:     2.4808     +112.7% ← 大幅増加
5K-6K:     2.3004     -7.3%
```

**観察**:
- 2K以降で急増
- 最終的に初期の2倍以上
- Perplexity低下と同期

**解釈**:
- 使用Codeが減少 → 量子化誤差増大
- Commitmentコストの増加
- **モデルがCodebook utilityと再構成品質でトレードオフ**

### 1.3 Codebook状態の健全性

```
全Quantizer共通:
  Norm mean: 6.24-6.30 (安定)
  Norm std:  0.67-0.73 (低分散)
  Dead codes: 0 (なし)

Q0 pairwise distances:
  Mean: 8.88
  Min:  5.25 (十分離れている)
  Max:  12.25
```

**観察**:
- **Codebookベクトルは健全**
- Dead codeなし（全て更新されている）
- ベクトル間距離も適切（近すぎず遠すぎず）

**重要な発見**:
❌ **Codebook崩壊ではない**
✅ **使用頻度の偏りが問題**

### 1.4 Progressive RVQの状況

**TensorBoardログ**:
- `train/rvq_perplexity_q0`: データあり
- `train/rvq_perplexity_q1-q7`: **データなし**

**解釈**:
- Q1-Q7のPerplexityが記録されていない
- 可能性1: Q1-Q7が有効化されていない
- 可能性2: Progressive RVQの実装問題

**検証が必要**:
```python
# trainer.py でのProgressive RVQの設定確認
current_step = ...
num_active = min(1 + current_step // config.progressive_steps, 8)
self.pipeline.decoder.rvq.set_num_active_quantizers(num_active)
```

---

## 2. 根本原因の特定

### 2.1 Perplexity低下のメカニズム

**標準的なRVQ学習**:
```
Step 1: 入力xに最も近いCode c_i を選択
Step 2: Commitment loss: ||x - c_i||^2
Step 3: Codebook更新（勾配降下）
```

**問題**:
- 勾配降下 → よく使われるCodeだけが更新される
- 使われないCode → 更新されない → 使われないまま
- **Rich gets richer**現象

**Phase 1での発生**:
1. 初期: ランダムなCodebook → 多様なCode使用 → Perplexity高
2. 学習進行: よく使われるCodeが入力に近づく
3. 収束: 少数のCode（~5個、Perplexity=5）のみ使用

### 2.2 なぜCodebookは健全なのに使われないのか

**矛盾**:
- Codebookベクトルは健全（Dead codeなし）
- しかしPerplexity低い（少数しか使われない）

**解明**:
- **全Codeは更新されている**（ノルム分布が健全）
- しかし**使用頻度が極端に偏っている**
- 例: Code 0-10が90%使用、Code 11-1023が10%使用

**原因**:
- Codebook更新 = 勾配降下（全パラメータ更新）
- Weight decayにより全Codeが同じように正則化
- → Dead codeにはならないが、使用頻度は偏る

### 2.3 Progressive RVQの問題

**設計意図**:
- 2000 stepごとに1 Quantizer追加
- Q0 (0-2K) → Q0+Q1 (2K-4K) → Q0+Q1+Q2 (4K-6K) → ...

**実際**:
- Q0のPerplexityのみ記録
- Q1-Q7のデータなし

**可能性1: 実装の問題**:
```python
# rvq.py で num_active_quantizers が正しく設定されていない？
# または、Perplexityの計算がQ0のみ？
```

**可能性2: Q1以降がほぼ使われていない**:
- Q0で既に十分量子化できている
- 残差が小さすぎてQ1以降が意味をなさない

---

## 3. 調査済みベストプラクティスの適用

### 3.1 DAC: EMA更新

**手法**:
```python
# Exponential Moving Average でCodebook更新
# 勾配降下ではなく、使用されたCodeの平均に置き換え

# Forward時
encodings = one_hot(indices, num_codes)  # (B*T, K)
cluster_size = encodings.sum(0)          # (K,) 各Codeの使用回数

# EMA更新
self.cluster_size = decay * self.cluster_size + (1-decay) * cluster_size
self.embed_avg = decay * self.embed_avg + (1-decay) * (encodings.T @ flatten)

# Codebook更新（勾配なし）
self.codebook = self.embed_avg / self.cluster_size.unsqueeze(1)
```

**メリット**:
- ✅ **使用頻度が自動的に均等化**
- ✅ 勾配の不安定性を回避
- ✅ DACで実証済み

**Phase 1への適用**:
- EMA decay: 0.99推奨
- 既存のCommitment lossと併用

### 3.2 DAC/EnCodec: Dead code replacement

**手法**:
```python
# 使用頻度が閾値以下のCodeを再初期化

if self.cluster_size[i] < threshold:
    # ランダム再初期化 or
    # 最も使われているCodeの近傍に配置
    self.codebook[i] = random_vector() * 0.01
```

**メリット**:
- ✅ **Dead codeの再活用**
- ✅ Codebook利用率向上

**Phase 1への適用**:
- threshold: 100 (全体の0.01%)
- 1000 stepごとにチェック

### 3.3 Improved VQGAN: Code balancing loss

**手法**:
```python
# 使用頻度を均等化するLoss

usage_prob = cluster_size / cluster_size.sum()  # (K,)
target_prob = 1.0 / K                            # 均等分布

balancing_loss = F.kl_div(
    usage_prob.log(),
    torch.full((K,), target_prob)
)
```

**メリット**:
- ✅ **明示的にバランス強制**
- ✅ Improved VQGANで効果実証

**Phase 1への適用**:
- Weight: 0.01-0.1

### 3.4 L2正規化（DAC）

**手法**:
```python
# 入力とCodebookをL2正規化
z_e = F.normalize(encodings, p=2, dim=-1)
codebook = F.normalize(self.codebook, p=2, dim=-1)

# 距離 = コサイン類似度
dist = z_e @ codebook.t()
```

**メリット**:
- ✅ **Codebook利用率向上**（Improved VQGAN実証）
- ✅ 訓練安定性

**デメリット**:
- ⚠️ StreamVC公式と異なる
- ⚠️ スケール情報消失（前回の問題）

**Phase 1への適用**:
- 慎重に検討（別Optionとして）

---

## 4. 推奨する改善策

### Option 1: EMA更新 + Dead code replacement（推奨★★★★★）

**実装**:
1. EMA更新の追加（rvq.py）
2. Dead code replacement（1000 stepごと）
3. 既存のCommitment lossは維持

**期待効果**:
- ✅ Perplexity: 5 → 15-20
- ✅ Codebook利用率向上
- ✅ 実証済み（DAC, EnCodec）

**リスク**: 🟢 低
- 学術的に確立
- 既存構造への影響最小

**実装コード例**:
```python
# rvq.py に追加

class ResidualVectorQuantizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ... existing code ...

        # EMA buffers
        self.register_buffer('cluster_size', torch.zeros(config.num_quantizers, config.codebook_size))
        self.register_buffer('embed_avg', torch.zeros(config.num_quantizers, config.codebook_size, config.dims))
        self.ema_decay = 0.99

    def ema_update(self, q_idx, encodings, flatten):
        # encodings: (B*T, K) one-hot
        # flatten: (B*T, C) input vectors

        cluster_size = encodings.sum(0)  # (K,)
        embed_sum = encodings.T @ flatten  # (K, C)

        # EMA更新
        self.cluster_size[q_idx] = (
            self.ema_decay * self.cluster_size[q_idx] +
            (1 - self.ema_decay) * cluster_size
        )
        self.embed_avg[q_idx] = (
            self.ema_decay * self.embed_avg[q_idx] +
            (1 - self.ema_decay) * embed_sum
        )

        # Codebook更新（no_grad）
        with torch.no_grad():
            n = self.cluster_size[q_idx].unsqueeze(1)
            self.codebooks[q_idx].data.copy_(
                self.embed_avg[q_idx] / (n + 1e-5)
            )

    def reset_dead_codes(self, q_idx, threshold=100):
        # 使用頻度が低いCodeを再初期化
        mask = self.cluster_size[q_idx] < threshold
        num_dead = mask.sum().item()

        if num_dead > 0:
            # ランダム再初期化
            self.codebooks[q_idx][mask] = torch.randn_like(self.codebooks[q_idx][mask]) * 0.01
            self.cluster_size[q_idx][mask] = 0
            self.embed_avg[q_idx][mask] = 0
```

---

### Option 2: Progressive RVQの修正（推奨★★★★☆）

**問題**:
- Q1-Q7のPerplexityデータなし
- Progressive RVQが機能していない可能性

**修正**:
1. Progressive RVQの実装を確認・修正
2. 各Quantizerのメトリクス記録
3. Progressive steps調整（2000 → 1000?）

**実装**:
```python
# trainer.py
current_step = self.global_step
num_active = min(1 + current_step // self.config.training.progressive_steps, 8)
self.pipeline.decoder.rvq.set_num_active_quantizers(num_active)

# 各QuantizerのPerplexityを記録
for q_idx in range(num_active):
    perplexity = compute_perplexity(codes[q_idx])
    self.log(f'train/rvq_perplexity_q{q_idx}', perplexity)
```

**期待効果**:
- ✅ 後段Quantizerの活用
- ✅ 表現力向上

**リスク**: 🟡 中
- Progressive RVQの実装依存

---

### Option 3: Commitment loss調整（推奨★★★☆☆）

**仮説**:
- Commitment cost 0.25が低すぎる
- Codebookが入力に近づきすぎる → 少数のCodeに収束

**修正**:
```yaml
commitment_cost: 0.5  # 0.25から増加
```

**期待効果**:
- ⚠️ Codebook更新を抑制
- ⚠️ 効果は限定的

**リスク**: 🟡 中
- 逆効果の可能性

---

### Option 4: L2正規化（検討中★★☆☆☆）

**前回の調査結果**:
- ✅ Codebook利用率向上（DAC実証）
- ❌ StreamVC公式と異なる
- ❌ スケール崩壊リスク

**判断**: ⚠️ **保留**
- まずOption 1, 2を試す
- 効果がなければ検討

---

## 5. 実装計画

### Phase 1-EMA: EMA更新版

**実装タスク**:
1. ✅ `rvq.py`: EMA buffers追加
2. ✅ `rvq.py`: `ema_update()` メソッド実装
3. ✅ `rvq.py`: `reset_dead_codes()` メソッド実装
4. ✅ `trainer.py`: EMA更新の呼び出し（毎step）
5. ✅ `trainer.py`: Dead code reset（1000 stepごと）
6. ✅ Config: `phase_1_ema.yaml` 作成

**検証計画**:
- 3K step学習
- Perplexity推移確認（目標: ≥10）
- Codebook使用頻度の分布確認

**成功基準**:
- ✅ Perplexity: ≥10（現在5→目標10以上）
- ✅ Perplexityの安定化（低下しない）
- ✅ Dead code率: <10%

---

## 6. まとめ

### Phase 1の問題

1. **Perplexity継続低下**: Winner-takes-all現象
2. **Progressive RVQ未機能**: Q1-Q7が使われていない可能性
3. **RVQ Loss増加**: 少数Code使用による量子化誤差増大

### 根本原因

❌ **Codebook崩壊ではない** → Codebookは健全
✅ **使用頻度の極端な偏り** → 少数Codeのみ使用
✅ **Progressive RVQの問題** → Q1-Q7が機能していない

### 推奨アプローチ

🎯 **Phase 1-EMA**: EMA更新 + Dead code replacement
- 学術的裏付けあり（DAC, EnCodec）
- 実装が明確
- リスク低

### 次のステップ

1. **即座**: Phase 1-EMA実装
2. **3K step後**: 結果検証
3. **成功なら**: さらに改善（GAN等）
4. **失敗なら**: Progressive RVQ修正 or L2正規化検討
