# StreamVC公式との比較分析

**日付**: 2025-11-07
**目的**: 現在の実装がStreamVC公式アーキテクチャ（Google, ICASSP 2024）と比較して変更する価値があるかを判断

---

## TL;DR

### 重要な発見

1. ✅ **Causal制約は既に実装済み**: `_causal_conv`で過去の情報のみ使用
2. ✅ **リアルタイム要件は満たしている**: 20ms chunk単位の処理
3. ⚠️ **公式はRVQを使用していない**: ミニバッチk-means + VQ（100個の中心）
4. ❌ **RVQ正規化は公式に記載なし**: f0の正規化のみ言及
5. 🔬 **DAC方式L2正規化は公式実装と異なる**

### 修正の価値判断

| 修正案 | 公式との整合性 | 期待効果 | リスク | 推奨度 |
|--------|-------------|---------|--------|--------|
| **Phase B (DAC L2正規化)** | ❌ 公式外 | Codebook利用率向上 | 中 | ⚠️ |
| **Phase 1回帰（シンプル化）** | ✅ 近い | 振幅安定化 | 低 | ✅ |
| **公式準拠（k-means VQ）** | ✅ 完全一致 | 公式と同等 | 高 | 🔬 |

---

## 1. StreamVC公式アーキテクチャ（Google, 2024）

### 1.1 全体構造

```
Input Audio (16kHz)
  ↓
Content Encoder (Causal Conv, no FiLM)
  ↓
Soft Speech Units (HuBERT派生, 100 centers)
  ↓ k-means VQ (RVQではない！)
  ↓
Decoder (SoundStream準拠, C=40, D=64)
  ├─ FiLM layers (speaker conditioning)
  ├─ f0 (whitened, running average)
  └─ energy
  ↓
Output Audio (16kHz, 20ms chunk)
```

### 1.2 重要な特徴

**Causal制約**:
- **全convolution層がcausal**
- 最小フレームサイズ: 320サンプル (20ms @ 16kHz)
- 2フレーム先読み → **60ms architectural latency**

**量子化手法**:
- **RVQではなく単純なVQ**（100個の中心）
- ミニバッチk-meansクラスタリング
- HuBERT派生の軟音声ユニット

**正規化**:
- **f0のみutterance-level正規化** (mean=0, std=1)
- ストリーミング時はrunning averages使用
- **RVQの正規化は記載なし**

**FiLM (Feature-wise Linear Modulation)**:
- デコーダのresidual units間に挿入
- スピーカー埋め込みからスケール・バイアス計算
- チャネルごとに `y = γ(speaker) * x + β(speaker)`

**勾配フロー制御**:
- デコーダからコンテンツエンコーダへの勾配を**遮断**
- コンテンツエンコーダは独立して学習

---

## 2. 現在の実装との比較

### 2.1 アーキテクチャレベルの違い

| 要素 | StreamVC公式 | 現在の実装 | 判定 |
|------|------------|-----------|------|
| **Causal制約** | ✅ 全Conv層 | ✅ `_causal_conv` | ✅ 一致 |
| **フレームサイズ** | 20ms (320 samples) | 20ms (320 samples) | ✅ 一致 |
| **量子化手法** | k-means VQ (100 centers) | **RVQ (8 quantizers, 1024 codebook)** | ❌ 大きく異なる |
| **Speaker conditioning** | FiLM layers | Linear projection + add | ⚠️ 方式が違う |
| **f0正規化** | Utterance-level (running avg) | Whitened | ✅ 類似 |
| **勾配制御** | Content encoderへの勾配遮断 | 未実装 | ❌ 欠落 |

### 2.2 RVQの採用による違い

**公式（k-means VQ）**:
```python
# 100個の中心のみ
centers = kmeans_clustering(soft_units, k=100)
quantized = nearest_neighbor(soft_units, centers)
# シンプル、高速
```

**現在の実装（RVQ）**:
```python
# 8段階の残差量子化、各1024コードブック
for i in range(8):
    residual = input - quantized
    quantized += vq(residual, codebook[i])
# 高品質、複雑
```

**トレードオフ**:
- 公式: **シンプル、低レイテンシ、モバイル最適化**
- RVQ: **高品質、表現力向上、計算コスト増**

### 2.3 正規化の違い

| 箇所 | StreamVC公式 | 現在の実装 | 備考 |
|------|------------|-----------|------|
| **f0正規化** | Utterance-level (mean, std) | Whitened (running avg) | ✅ 類似 |
| **VQ入力正規化** | **記載なし** | Z-score (mean=0, std=1) | ⚠️ 独自実装 |
| **Codebook正規化** | **記載なし** | なし | - |

**重要**: 公式論文には**VQ入力の正規化について言及がない** = おそらく正規化していない可能性

---

## 3. Phase B（DAC L2正規化）の妥当性再評価

### 3.1 公式との整合性

❌ **StreamVC公式はL2正規化を使用していない**

**理由**:
- 公式は**SoundStreamアーキテクチャ準拠**
- SoundStream自体も**VQ入力の正規化なし**
- DACはImproved VQGANの派生（画像生成由来）

### 3.2 DAC方式の利点は依然有効か？

✅ **学術的価値はある**（DAC, Improved VQGAN）
⚠️ **しかしStreamVCの目的と異なる**

**DACの目的**:
- ハイファイ音声コーデック
- 高圧縮率
- **品質最優先**

**StreamVCの目的**:
- **リアルタイム性**
- **低レイテンシ**
- モバイル動作
- 品質は「十分」であれば良い

### 3.3 判断

Phase B（DAC L2正規化）は:
- ✅ Codebook利用率向上は期待できる
- ❌ **公式StreamVCの方針と異なる**
- ⚠️ レイテンシ増加の可能性（正規化計算コスト）
- ⚠️ **本来の目的（リアルタイムVC）から逸脱するリスク**

---

## 4. 公式準拠への修正（k-means VQ採用）

### 4.1 k-means VQへの変更

**実装イメージ**:
```python
# RVQを削除、シンプルなVQに変更
class SimpleVQ(nn.Module):
    def __init__(self, num_centers=100, dim=64):
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(num_centers, dim))

    def forward(self, x):
        # x: (B, T, C)
        # 最近傍探索（正規化なし）
        distances = torch.cdist(x, self.codebook)  # (B, T, num_centers)
        indices = distances.argmin(dim=-1)         # (B, T)
        quantized = F.embedding(indices, self.codebook)  # (B, T, C)

        # STE
        quantized = x + (quantized - x).detach()

        return quantized, indices
```

**メリット**:
- ✅ 公式と完全一致
- ✅ シンプル、デバッグ容易
- ✅ 低レイテンシ
- ✅ モバイル動作しやすい

**デメリット**:
- ❌ Codebook数が少ない（100 vs 8192）
- ❌ 表現力が低い
- ❌ 音質低下の可能性

### 4.2 段階的移行プラン

**Option: Phase C（公式準拠VQ）**

1. RVQをシンプルVQ（100 centers）に変更
2. 正規化を削除（公式に合わせる）
3. FiLM layersの実装検討
4. 勾配遮断の実装

**リスク**: ⚠️ 高
- RVQからの大幅な変更
- 音質低下の可能性
- 学習の不安定化

---

## 5. Phase 1回帰（シンプル化）の妥当性再評価

### 5.1 公式との関係

✅ **Phase 1は公式に最も近い**

**Phase 1の構造**:
```
Content Encoder
  ↓
(正規化)
  ↓
RVQ (8 quantizers)
  ↓
out_proj (単純な1x1 conv)
  ↓
Audio
```

**公式との類似点**:
- シンプルな構造
- 正規化は最小限（f0のみ）
- 追加の複雑な層なし

**公式との相違点**:
- RVQ vs k-means VQ（量子化手法が違う）
- Codebook数が多い（8192 vs 100）

### 5.2 Phase 1の優位性

✅ **実証済みの安定性** (Audio RMS 0.080)
✅ **シンプル** (勾配が直接届く)
✅ **公式の「シンプルさ重視」の思想に合致**

⚠️ **Perplexity低い問題はある** (5-8)
- しかしこれは**RVQ特有の問題**
- 公式はk-means VQ（100 centers）なので比較不可

---

## 6. 最終推奨

### 6.1 優先度付き修正案

#### 🥇 優先度1: Phase 1回帰（最も安全かつ合理的）

**実装**:
- post_scale, post_bias, final_convを削除
- `rvq → out_proj`のみ
- RMS Lossも削除（Phase 1に倣う）

**理由**:
- ✅ 実証済みの振幅安定性
- ✅ 公式の「シンプルさ」思想に合致
- ✅ 低リスク
- ✅ Causal制約、リアルタイム性を維持

**デメリット**:
- Perplexity低下（5-8）→ モスキートーン
- しかしこれは**次のフェーズで対処可能**

#### 🥈 優先度2: Phase 1 + Codebook利用率対策

**実装**:
- Phase 1ベース
- EMA更新追加（DAC, EnCodecで使用）
- Dead code replacementの実装

**理由**:
- ✅ Phase 1の安定性を保持
- ✅ Perplexity改善を図る
- ✅ 公式外だが、一般的な手法

#### 🥉 優先度3: Phase C（公式完全準拠）

**実装**:
- RVQをk-means VQ（100 centers）に変更
- 正規化削除
- FiLM layers実装

**理由**:
- ✅ 公式と完全一致
- ❌ **大幅な変更、高リスク**
- ❌ 音質低下の可能性

### 6.2 DAC方式（Phase B）の扱い

**判断**: ⚠️ **保留または非推奨**

**理由**:
- StreamVC公式の思想（シンプル、リアルタイム）と異なる
- DACは音声コーデック特化（ハイファイ音質優先）
- L2正規化の計算コスト増

**例外的に検討する場合**:
- Perplexity問題が深刻で他の方法で解決できない
- リアルタイム性を犠牲にしても音質を優先したい場合

---

## 7. 修正方針の決定

### 推奨アプローチ: Phase 1回帰 + 段階的改善

**Step 1: Phase 1回帰（即座に実施可能）**
```python
# decoder.py
x_for_rvq = normalize(self.pre_rvq_conv(x))
quantized, rvq_loss, codes = self.rvq(x_for_rvq)
audio = self.out_proj(quantized.transpose(1, 2)).squeeze(1)
# ↑ post_scale, final_convなし
```

**Step 2: Perplexity改善（Phase 1が成功したら）**
- EMA更新の追加
- Dead code replacementの実装
- Progressive RVQの調整

**Step 3: 音質改善（Perplexityが改善したら）**
- GAN discriminatorの追加（Phase C）
- Multi-scale discriminator
- Feature matching loss

### なぜこの順序か

1. **まず振幅崩壊を解決**（Phase 1回帰）
   - これが最優先課題
   - 低リスク、実証済み

2. **次にCodebook利用率改善**（EMA等）
   - モスキートーン対策
   - 中リスク、一般的手法

3. **最後に音質向上**（GAN等）
   - 振幅とCodebookが安定してから
   - 高リスク、複雑

---

## 8. Causal制約の確認

### 8.1 現在の実装

✅ **Causal制約は既に実装済み**

```python
# src/streamvc/modules/decoder.py:14-17
def _causal_conv(x: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
    padding = (conv.kernel_size[0] - 1) * conv.dilation[0]
    x = nn.functional.pad(x, (padding, 0))
    return conv(x)
```

**使用箇所**:
- ResidualBlock (decoder.py:30)
- UpsampleBlock (decoder.py:52)

### 8.2 リアルタイム性の確認

✅ **20ms chunk処理を想定**

```yaml
# configs/phase_a_v2_fixed.yaml:7-8
frame_ms: 20.0
lookahead_frames: 3
```

**計算**:
- フレーム長: 20ms
- 先読み: 3フレーム = 60ms
- **Latency: 60ms** (公式と同じ)

### 8.3 結論

✅ **Causal制約、リアルタイム性は既に満たしている**
- RVQ正規化の変更はこれらを損なわない
- Phase 1回帰もCausal制約を維持

---

## 9. まとめ

### 質問への回答

> そもそも学習の目的として、リアルタイムのAny -> Anyであるため、
> 過去の情報しか学習に使うことができない構造は変わらないよな？

**回答**: ✅ **はい、変わりません**
- Causal制約は既に実装済み
- 全ての修正案でCausal制約を維持

> streamVCの公式のアーキテクチャと比較して、
> 優位に差が出そうな変更ならやる価値はあるかも

**回答**: ⚠️ **Phase B（DAC L2正規化）は公式と異なる方向**

**代わりに推奨**:
- ✅ **Phase 1回帰**: 公式の「シンプルさ」思想に最も近い
- ✅ 実証済みの振幅安定性
- ✅ Causal制約、リアルタイム性を完全維持
- ⚠️ Perplexity問題は次のステップで対処

### 最終推奨

🎯 **Phase 1回帰を即座に実施**
- post_scale, final_convを削除
- RMS Lossも削除
- シンプルに `rvq → out_proj`

📊 **成功したら段階的改善**
- EMA更新でPerplexity改善
- GAN discriminatorで音質向上

❌ **Phase B（DAC L2正規化）は非推奨**
- 公式StreamVCの方針と異なる
- リアルタイムVC用途に不適切
