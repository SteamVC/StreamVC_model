# StreamVC開発の方針決定マトリクス

**日付**: 2025-11-07
**状況**: Phase 1はモスキートーン問題未解決、公式実装は非公開、選択を迫られている

---

## 現状の整理

### 判明した事実

1. ❌ **Google公式実装は非公開**
2. ❌ **非公式実装2つとも不完全**（WIP、学習済みモデルなし）
3. ❌ **RVQの実装詳細は論文に記載なし**
4. ✅ **Phase 1は振幅安定**（Audio RMS 0.080）
5. ❌ **Phase 1はモスキートーン問題あり**（Perplexity 5-8）

### 現在の立ち位置

```
あなたの実装：
- Causal制約: ✅ 実装済み
- RVQ: ✅ 8 quantizers, 1024 codebook（公式より高表現力）
- 振幅安定: ✅ Phase 1で達成
- モスキートーン: ❌ 未解決（Perplexity低下）

公式StreamVC（Google）:
- 実装: ❌ 非公開
- 詳細: ❌ 論文に不足（k-means VQの詳細不明）
- 音質: ✅ 論文のデモは高品質
- 再現性: ❌ 非公式実装も不完全
```

---

## 選択肢の詳細分析

### Option A: 公式完全準拠を目指す

**実装内容**:
1. RVQ → k-means VQ（100 centers）に変更
2. FiLM layers実装（スピーカーconditionin）
3. 勾配遮断実装
4. その他の隠された実装詳細の推測

**メリット**:
- ✅ 学術的に正当
- ✅ 論文の結果を再現できる（理論上）

**デメリット**:
- ❌ **実装詳細が不明**（論文に書かれていない部分が多い）
- ❌ **非公式実装も不完全で参考にならない**
- ❌ **RVQからk-means VQへの大幅変更**（表現力低下）
- ❌ **k-meansの実装詳細不明**（ミニバッチサイズ、更新頻度等）
- ❌ **高リスク**（動かない可能性あり）

**リスク評価**: 🔴 **高リスク**
- 推測実装になる
- 公式と異なる結果になる可能性大
- デバッグ困難

---

### Option B: 自前でRVQベースの改善を探索

**実装内容**:
1. Phase 1ベースに戻る（振幅安定化）
2. Perplexity改善を段階的に試行
   - EMA更新
   - Dead code replacement
   - Commitment loss調整
   - Progressive RVQの最適化
3. 音質改善
   - GAN discriminator追加
   - Feature matching loss
   - Multi-scale discriminator
4. RVQの表現力を最大活用

**メリット**:
- ✅ **段階的改善**（各ステップで検証可能）
- ✅ **RVQの高表現力を活用**（8192 codes vs 100 centers）
- ✅ **実装済みの基盤を活用**
- ✅ **デバッグ容易**（既知の動作から改善）
- ✅ **独自の貢献**（公式と異なるアプローチ）

**デメリット**:
- ⚠️ **試行錯誤が必要**
- ⚠️ **公式を超えられる保証なし**
- ⚠️ **時間がかかる可能性**

**リスク評価**: 🟡 **中リスク**
- 段階的検証で早期発見可能
- Phase 1の成功実績あり

---

### Option C: ハイブリッド（部分的に公式準拠）

**実装内容**:
1. RVQは維持（表現力を保持）
2. FiLM layers実装（公式準拠）
3. 勾配遮断実装（公式準拠）
4. その他は自前で改善

**メリット**:
- ✅ 公式の良い部分を取り入れ
- ✅ RVQの表現力は維持
- ✅ 比較的低リスク

**デメリット**:
- ⚠️ ハイブリッドの一貫性が不明
- ⚠️ FiLM layersの効果が不明

**リスク評価**: 🟡 **中リスク**

---

## 各選択肢の成功確率と期待効果

| 選択肢 | 成功確率 | 振幅安定 | モスキートーン解決 | 音質 | 期間 |
|--------|---------|---------|------------------|------|------|
| **A: 公式完全準拠** | 20% | ❓ | ❓ | ❓ | 2-4週 |
| **B: 自前RVQ改善** | 70% | ✅ | 🟡（段階的） | 🟢（期待大） | 2-6週 |
| **C: ハイブリッド** | 50% | ✅ | 🟡 | 🟡 | 2-4週 |

**成功確率の根拠**:
- A: 実装詳細不明、推測実装になるため低い
- B: Phase 1成功実績、段階的検証可能、実装ベースありで高い
- C: 中間的

---

## 推奨アプローチ: Option B（自前RVQ改善）

### 理由

1. **実装詳細不明のリスク回避**
   - 公式実装非公開
   - 論文に不足情報多数
   - 非公式実装も不完全

2. **既存の成功実績を活用**
   - Phase 1で振幅安定化達成
   - Causal制約、リアルタイム性実装済み
   - デバッグ可能な基盤あり

3. **RVQの優位性**
   - 8192 codes（公式100の81倍）
   - 高表現力
   - DAC等の先行研究あり

4. **段階的改善の利点**
   - 各ステップで検証可能
   - 早期に問題発見
   - リスク最小化

---

## Option B: 具体的な実装プラン

### Phase 1: 振幅安定化（即座）

**実装**:
```python
# post_scale, final_convを削除
# RMS Lossも削除
# シンプルに rvq → out_proj
```

**目標**:
- ✅ Audio RMS: 0.075 ± 20%
- ✅ Out_proj安定化
- ⚠️ Perplexity: 5-8（問題は次で対処）

**期間**: 即座（既存コード削除のみ）

---

### Phase 2: Perplexity改善（1-2週）

**実装A: EMA更新**
```python
# RVQにEMA（Exponential Moving Average）追加
# DACの実装参考
self.register_buffer('cluster_size', torch.zeros(num_codes))
self.register_buffer('embed_avg', codebook.clone())

# 更新
self.cluster_size.mul_(decay).add_(1 - decay, encodings.sum(0))
self.embed_avg.mul_(decay).add_(1 - decay, flatten @ encodings)
self.codebook.data.copy_(self.embed_avg / self.cluster_size.unsqueeze(1))
```

**実装B: Dead code replacement**
```python
# 使用頻度が低いコードを置換
if self.cluster_size[i] < threshold:
    self.codebook[i] = random_reinit()
```

**実装C: Commitment loss調整**
```python
# 現在: commitment_cost = 0.5
# 試す: 0.25, 0.1, 1.0
```

**目標**:
- ✅ Perplexity: 8-15
- ✅ Codebook利用率向上
- ✅ モスキートーン軽減

---

### Phase 3: 音質改善（2-3週）

**実装A: GAN discriminator**
```python
# Multi-scale discriminator（MelGAN方式）
class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        self.discriminators = nn.ModuleList([
            Discriminator(scale=1),
            Discriminator(scale=2),
            Discriminator(scale=4),
        ])
```

**実装B: Feature matching loss**
```python
# Discriminatorの中間特徴を一致させる
feature_loss = sum(
    F.l1_loss(real_feat, fake_feat)
    for real_feat, fake_feat in zip(real_features, fake_features)
)
```

**目標**:
- ✅ 音質向上
- ✅ モスキートーン完全解消
- ✅ 自然な音声生成

---

### Phase 4: 最適化（1-2週、オプション）

**実装**:
- Progressive RVQの調整
- Loss weightの最適化
- Architectureの微調整

---

## 代替案: Option Cの場合

もしOption C（ハイブリッド）を選ぶ場合：

### 追加実装: FiLM layers

```python
# Decoderに追加
class FiLMLayer(nn.Module):
    def __init__(self, channels, speaker_dim):
        super().__init__()
        self.scale_net = nn.Linear(speaker_dim, channels)
        self.bias_net = nn.Linear(speaker_dim, channels)

    def forward(self, x, speaker_emb):
        # x: (B, C, T)
        # speaker_emb: (B, D)
        scale = self.scale_net(speaker_emb).unsqueeze(-1)  # (B, C, 1)
        bias = self.bias_net(speaker_emb).unsqueeze(-1)    # (B, C, 1)
        return scale * x + bias
```

### 追加実装: 勾配遮断

```python
# pipeline.py
units = self.content_encoder(source_audio)
units = units.detach()  # 勾配遮断
```

**期待効果**:
- スピーカーconditioningの改善
- Content encoderの独立学習

**リスク**:
- 効果が不明（公式実装との違いが大きい）

---

## 最終推奨

🎯 **Option B: 自前RVQ改善を段階的に実施**

**理由**:
1. 公式実装が非公開で詳細不明
2. Phase 1の成功実績あり
3. 段階的検証でリスク最小化
4. RVQの高表現力を活用

**次のアクション**:
1. ✅ **即座**: Phase 1回帰（振幅安定化）
2. 📊 **1-2週後**: Perplexity改善（EMA, dead code等）
3. 🎵 **3-5週後**: 音質改善（GAN等）

**成功基準**:
- Audio RMS: 0.075 ± 20%
- Perplexity: ≥10
- 主観評価: モスキートーンなし、自然な音声

---

## 次のステップ

どの選択肢を選びますか？

1. **Option B推奨: 自前RVQ改善**
   → Phase 1回帰を即座に実施

2. **Option A検討: 公式完全準拠**
   → さらなる論文調査、リスク高

3. **Option C検討: ハイブリッド**
   → FiLM layers等の部分実装

4. **保留・追加調査**
   → 他の類似研究の調査
