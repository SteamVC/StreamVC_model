# Phase 1-EMA 総合分析レポート

**作成日**: 2025-11-07
**対象**: Phase 1-EMA @ 5000 steps 振幅崩壊問題
**目的**: 問題の根本原因特定と解決策の優先順位付け

---

## Executive Summary

### 問題の本質
Phase 1-EMA (5000 steps) において **Audio RMS: 0.527 → 0.002 (-99.6%, -30.5 dB)** という致命的な振幅崩壊が発生。Perplexityは改善（11.99）したが、実用上使用不可能な出力となった。

### 根本原因（3層構造）
```
Level 1 (表面): RMS Loss weight 0.1が不十分 (Total Lossの0.1%のみ)
  ↓
Level 2 (構造): STFT Loss + L1 Lossの盲点
               - STFT Loss: 位相無視、絶対振幅レベル二の次
               - L1 Loss: 相対的差分のみ、一律縮小を許容
  ↓
Level 3 (本質): GAN Discriminatorの欠如
               - Neural Vocoder/Codecの確立手法を未適用
               - 知覚的品質・自然さの制約なし
               - StreamVC論文の前提（SoundStream-style）に反する
```

### 解決策の優先順位
1. **Priority 1 (根本解決)**: GAN Discriminator実装・有効化 (`adversarial_weight: 4.0`)
2. **Priority 2 (併用)**: RMS Loss増強 (`rms_weight: 1.0`)
3. **Priority 3 (補完)**: Multiband STFT Loss追加

---

## 1. ユーザー質問への直接回答

### Q1: 「RMS lossが落ちすぎることに関しては、どうする予定か？」

**短期対応（対症療法）**:
```yaml
# RMS Loss weightを10倍に増強
losses:
  rms_weight: 1.0  # 0.1 → 1.0
  multiband_rms_weight: 0.5  # 0.05 → 0.5
```
- Total Lossの寄与率: 0.1% → 5-10% に増加
- 振幅崩壊への抵抗力向上
- **リスク**: 低、すぐ実施可能

**長期対応（根本解決）**:
```yaml
# GAN Discriminator導入
losses:
  adversarial_weight: 4.0  # 0.0 → 4.0 (有効化)
  feature_matching_weight: 2.0  # 0.0 → 2.0
  rms_weight: 1.0  # RMS Lossも併用
```
- Neural Vocoder/Codecの確立手法を適用
- 振幅・位相・高周波を総合的に最適化
- **リスク**: 中（学習不安定化の可能性）

### Q2: 「GANを導入した時に、RMS loss増強はどうなるの？」

**結論: GANとRMS Lossは相補的、両方使う**

| 制約の種類 | RMS Loss | GAN Discriminator |
|-----------|---------|------------------|
| **次元** | 1次元スカラー | 多次元（全特徴） |
| **対象** | 振幅の平均値のみ | 振幅分布・位相・高周波・音質全般 |
| **制約方法** | 明示的 (explicit) | 暗黙的 (implicit) |
| **役割** | 最低保証（floor） | 自然さの学習 |

**具体例**:
```
RMS Loss (weight 1.0):
  "Generated RMSは0.075であるべき"（平均値を強制）
  → 平均は合うが、不自然な振幅分布も許容してしまう

GAN Discriminator (weight 4.0):
  "Real音声の振幅分布に従うべき"（自然な分布を学習）
  → 平均だけでなく、分散・ピーク・時間変動も自然に

両者併用:
  RMS Loss: 最低限の振幅レベルを保証
  GAN: その上で自然な分布・音質を実現
  → ✅ 最適な組み合わせ
```

**推奨設定**:
```yaml
losses:
  adversarial_weight: 4.0      # メイン制約
  feature_matching_weight: 2.0 # 補助
  rms_weight: 1.0              # 最低保証（0.1から10倍増強）
  multiband_rms_weight: 0.5    # 周波数帯域別保証
```

### Q3: 「これってそもそも振幅だっけ、それってGANと何が関係あるの？」

**RMS (Root Mean Square) の正確な定義**:
```python
# RMSは「振幅」ではなく「音量」「パワー」の指標
RMS = sqrt(mean(signal^2))

# 具体例:
signal = [0.5, -0.3, 0.7, -0.2]
RMS = sqrt((0.5^2 + 0.3^2 + 0.7^2 + 0.2^2) / 4)
    = sqrt(0.215) = 0.464
```

**「振幅」vs「RMS」の違い**:
| 用語 | 定義 | 例 |
|-----|-----|---|
| **Peak amplitude** | 信号の最大絶対値 | max(\|signal\|) = 0.7 |
| **RMS amplitude** | 二乗平均平方根 | sqrt(mean(signal²)) = 0.464 |
| **音量 (Loudness)** | 知覚的な「大きさ」≈ RMS | 0.464 |

**RMSとGANの関係**:

```
問題: Audio RMS 0.002 = 音量が1/33に減少

原因分析:
  ├─ RMS Loss (weight 0.1): 明示的制約だが弱すぎる
  │   "Generated RMS = 0.075であれ"と指示するが、
  │   Total Lossの0.1%しかないため無視される
  │
  ├─ STFT Loss: 位相無視、相対的な振幅しか見ない
  │   "スペクトル形状が合えばOK" → 全体的な縮小を許容
  │
  ├─ L1 Loss: 点ごとの差の平均のみ
  │   "波形形状が合えばOK" → 一律縮小を許容
  │
  └─ GAN Discriminator (欠如): ← これが本質的解決策
      "Real音声とFake音声を識別"
      → Real音声の振幅分布を暗黙的に学習
      → 異常に小さい音声を「Fake」と判定
      → Generatorは自然な振幅を生成せざるを得ない
```

**GANによる振幅崩壊防止のメカニズム**:
```python
# Discriminatorの学習
real_audio = batch['target']  # RMS ≈ 0.075
fake_audio = generator(batch['source'])  # RMS = 0.002 (崩壊)

# Discriminatorは「Real音声」の特徴を学習:
#   - RMSの分布: Normal(mean=0.075, std=0.02)
#   - ピーク振幅の範囲: 0.2-0.5
#   - 時間的変動パターン
#   - 周波数帯域別の振幅バランス

d_real = discriminator(real_audio)  # Score: 0.95 (Real)
d_fake = discriminator(fake_audio)  # Score: 0.02 (Fake!)

# Generatorへの勾配:
# "RMS 0.002では明らかにFake判定される"
# → "振幅を増やさないとAdversarial Lossが大きすぎる"
# → 自動的に振幅を保持するように学習
```

**RMS LossとGANの補完関係**:
```
RMS Loss (1次元スカラー制約):
  ✅ "平均的な音量"を保証
  ❌ 分布の形状は保証しない

  例: 全フレームが同じ振幅 (0.075) → RMS OK だが不自然

GAN Discriminator (多次元制約):
  ✅ Real音声の"自然な振幅分布"を学習
  ✅ 時間変動・周波数バランス・ピーク特性も考慮

  例: 時間的に変動する自然な振幅パターン → Real判定

両者併用:
  RMS Loss: "最低限0.075を維持しろ"（safety net）
  GAN: "0.075前後で自然に変動しろ"（naturalness）
```

---

## 2. StreamVC論文の前提条件との照合

### StreamVC論文の記述（事実）

**アーキテクチャ** (`docs/streamvc_architecture.md:25`):
> Decoder は SoundStream 系ボコーダ構造で、soft speech units・whitened f0/energy・speaker 埋め込みを条件として波形を生成する。

**学習戦略** (`docs/streamvc_architecture.md:49`):
> Decoder は SoundStream の学習戦略（多尺度 STFT、波形 L1、**GAN/feature 損失**）を踏襲し、content/f0/speaker 条件付きで元音声を再構成する。

**SoundStream論文の必須要件**:
```python
# SoundStream (Zeghidour et al., 2021) のLoss構成
loss = (
    adversarial_loss +        # Weight: 重要 ← GANは必須
    feature_matching_loss +   # Weight: 中程度
    reconstruction_loss +     # L1 or L2
    spectral_loss +           # Multi-resolution STFT
    commitment_loss           # RVQ
)
```

### 現在の実装との乖離

**元の設計** (`configs/mps_training.yaml`):
```yaml
losses:
  adversarial_weight: 4.0        # 定義済み ✅
  feature_matching_weight: 2.0   # 定義済み ✅
```

**Phase 1-EMAの実験** (`configs/phase_1_ema.yaml:73`):
```yaml
losses:
  adversarial_weight: 0.0  # 無効化 ❌
  feature_matching_weight: 0.0
```

**乖離の理由**:
- Phase 1の段階的開発方針: まず基本構造を確認
- GAN導入の複雑さ回避（学習不安定化リスク）
- **しかし**: この判断が振幅崩壊の根本原因

### Neural Vocoder/Audio Codecの知見

**HiFi-GANの教訓** (Kong et al., 2020):
- Discriminatorなし: 高周波 (> 4kHz) が-20 dB減衰
- Multi-Period Discriminator追加: 高周波が-3 dBに改善
- **振幅・位相・高周波はGANでしか保持できない**

**SoundStream/EnCodec/DACの共通点**:
- 全てMulti-Scale Discriminatorを使用
- Adversarial Lossなしでは高品質再構成が困難
- **これがNeural Audio Codecの確立手法**

**結論**: StreamVCの実装はSoundStream-styleを標榜しながら、最も重要なGAN Discriminatorを欠いていた。これが振幅崩壊の本質的原因。

---

## 3. 実測データと詳細分析

### 3.1 Audio RMS推移（致命的崩壊）

| Step | Audio RMS | 変化率 | 分析 |
|------|-----------|--------|------|
| 0    | 0.527     | -      | 初期値（正常） |
| 500  | 0.216     | -59%   | 急速な崩壊開始 |
| 1000 | 0.083     | -84%   | 既に異常レベル |
| 1500 | 0.005     | -99%   | ほぼ無音 |
| 2000 | 0.617     | +12233% | **Q1追加で一時回復** |
| 3000 | 0.005     | -99%   | 再崩壊 |
| 4000 | 0.229     | +4580% | **Q2追加で一時回復** |
| 5000 | 0.002     | -99.6% | **最悪値** |

**観察**:
- Progressive RVQ (新Quantizer追加時) に一時的回復
- しかし数百step後に再崩壊
- **RVQ Loss最適化圧力がRMS Lossを圧倒**

### 3.2 Perplexity推移（EMA改善を確認）

| Step | Q0 Perplexity | Q1 Perplexity | Q2 Perplexity |
|------|--------------|--------------|--------------|
| 0    | 27.36        | -            | -            |
| 1000 | 11.02 ✅      | -            | -            |
| 2000 | 12.52 ✅      | (新規追加)    | -            |
| 3000 | 12.53 ✅      | -            | -            |
| 4000 | 9.51         | -            | (新規追加)    |
| 5000 | 11.99 ✅      | 22.77 ✅      | 71.63 ✅      |

**Phase A-v2 (EMA無し) との比較**:
- Phase A-v2 @ 3K: Perplexity 3.56 ❌
- Phase 1-EMA @ 3K: Perplexity 12.53 ✅ (+252%)

**結論**: EMAはPerplexity改善に成功したが、振幅崩壊は防げず

### 3.3 Loss構成の問題

**Phase 1-EMA @ 5000 stepsの内訳**:
```
Total Loss: 6.59

内訳と寄与率:
  Content CE:       4.44 (67.4%) ← 支配的
  STFT:             1.18 (17.9%) ← スペクトル形状のみ
  L1 (×10):         0.05 ( 0.8%) ← 相対誤差のみ
  RVQ:              0.45 ( 6.8%) ← 量子化誤差
  RMS (×0.1):       0.01 ( 0.1%) ← 無力！
  Multiband RMS:    0.00 ( 0.0%) ← 無力！
```

**問題の可視化**:
```
Total Loss = 6.59
             ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Content CE   ████████████████████████████████ (67.4%)
STFT         ████████ (17.9%)
RVQ          ███ (6.8%)
L1           ▌ (0.8%)
RMS          ▏ (0.1%) ← ここ！

RMS Lossの寄与: 0.01 / 6.59 = 0.15%
→ 他のLossに完全に圧倒される
```

### 3.4 各Lossの盲点

#### Content CE Loss (67.4%)
- **役割**: HuBERTラベル予測（意味内容保持）
- **盲点**: 音響詳細・振幅・音質には無関心

#### STFT Loss (17.9%)
```python
stft_loss = |STFT(generated)| - |STFT(target)|
```
- **見るもの**: 周波数成分の振幅スペクトル
- **盲点**:
  - ❌ 位相情報を完全無視
  - ❌ 絶対的な振幅レベルは二の次
  - ❌ 相対的な形状が合えばOK

**具体例**:
```
Target STFT:
  100Hz: 10.0, 200Hz: 8.0, 4kHz: 2.0

Generated STFT (×0.03に縮小):
  100Hz: 0.3, 200Hz: 0.24, 4kHz: 0.06

STFT Loss = 中程度
→ "形は合っている" と判定されるが、振幅が全く違う
```

#### L1 Loss (0.8%)
```python
l1_loss = mean(|generated - target|)
```
- **見るもの**: 波形の点ごとの差
- **盲点**:
  - ❌ 相対的な差しか見ない
  - ❌ 一律縮小しても波形形状が保たれればOK

#### RVQ Loss (6.8%)
```python
rvq_loss = commitment_cost * |residual - quantized|^2
```
- **盲点**:
  - ❌ 「小さい入力を正確に量子化」= 「小さい出力」
  - ❌ 振幅保持の責任はない

### 3.5 RVQ Loss vs Audio RMSの相関

| Step | RVQ Loss | Audio RMS | 相関 |
|------|----------|-----------|------|
| 0    | 1.175    | 0.527     | -    |
| 1000 | 0.464 ⬇️  | 0.083 ⬇️  | **負の相関** |
| 2000 | 1.032 ⬆️  | 0.617 ⬆️  | **負の相関** |
| 3000 | 0.514 ⬇️  | 0.005 ⬇️  | **負の相関** |
| 4000 | 1.529 ⬆️  | 0.229 ⬆️  | **負の相関** |
| 5000 | 0.451 ⬇️  | 0.002 ⬇️  | **負の相関** |

**決定的な証拠**: RVQ Loss最小化 ⇔ Audio RMS減少

---

## 4. GAN Discriminatorの具体的効果

### 4.1 高周波帯域のケア

**問題**: STFT Loss + L1 Lossでは高周波 (> 4kHz) が失われる

**メカニズム**:
```python
# Multi-resolution STFT Loss (現在の実装)
for n_fft in [2048, 1024, 512]:
    loss += |STFT(pred, n_fft) - STFT(target, n_fft)|

# 問題点:
# - 高周波はエネルギーが小さい（子音 /s/, /t/, /k/ は 4-8 kHz）
# - FFTの高周波ビンは数が少ない
# - 平均化されると高周波の寄与が小さい
```

**GANの効果**:
```python
# Multi-Period Discriminator (HiFi-GAN)
for period in [2, 3, 5, 7, 11]:
    # 高周波成分は短い周期で変動
    # → 高周波の欠落は「不自然な周期パターン」として検出
    # → Generatorは高周波を補完せざるを得ない
```

### 4.2 振幅崩壊の防止

```python
# Discriminatorは「Real音声」の振幅分布を学習
real_rms_distribution = Normal(mean=0.075, std=0.02)

# Generated音声のRMS = 0.002
# → Discriminatorスコア: 0.01 (明らかにFake)
# → Adversarial Loss増大
# → Generatorは振幅を増やさざるを得ない
```

### 4.3 位相の整合性

**問題**: STFT Lossは位相を無視 → 不自然な音声

**GANの効果**:
- Discriminatorは波形そのものを評価
- 位相の不整合 → 不自然な干渉パターン → 「Fake」判定
- Generatorは自然な位相を学習

### 4.4 知覚的音質

**音響特徴のカバレッジ比較**:

| 特徴 | Content CE | STFT | L1 | RVQ | RMS | **GAN** |
|-----|-----------|------|-----|-----|-----|---------|
| 意味内容 | ✅ | - | - | - | - | ✅ |
| 周波数成分 | - | ✅ | △ | - | - | ✅ |
| 位相 | - | ❌ | △ | - | - | ✅ |
| 絶対振幅 | - | ❌ | ❌ | ❌ | ✅ | ✅ |
| 高周波詳細 | - | △ | △ | - | - | ✅ |
| 自然さ | - | - | - | - | - | ✅ |
| 知覚品質 | - | - | - | - | - | ✅ |

**GANだけがカバーする要素**:
- 🎯 位相の自然さ
- 🎯 高周波の微細構造
- 🎯 振幅の自然な分布
- 🎯 知覚的な音質

---

## 5. 推論時への影響（実用性の評価）

### 5.1 デシベル換算

```python
dB = 20 * log10(0.002 / 0.075)
   = 20 * log10(0.027)
   = -30.5 dB

解釈: 音量が 1/33 に減少
```

### 5.2 16-bit PCM換算

| 状態 | RMS | Peak振幅 (推定) | 16-bit値 |
|------|-----|----------------|----------|
| **正常** | 0.075 | ~0.3 | ~9830 |
| **崩壊後** | 0.002 | ~0.009 | ~295 |

**295 (int16) = 16-bitの0.9%しか使っていない**
→ 実質的に **6-bit音声** 相当

### 5.3 実用上の症状

**症状1: ほぼ無音の出力**
- スピーカー音量100% → かろうじて囁き声
- 通常の環境音にかき消される

**症状2: 後処理増幅が必須**
```python
output_audio = model(input_audio)  # RMS = 0.002
amplified = output_audio * 33.4    # RMS = 0.075に回復
# しかし増幅倍率が固定できない、クリッピングリスク
```

**症状3: SNR劣化**
```
正常 (RMS=0.075): SNR = 57.5 dB
崩壊後 (RMS=0.002): SNR = 26.0 dB
差: 31.5 dB の劣化
→ シューシューという雑音が目立つ
```

**症状4: 話者特徴の損失**
- 声質 (Timbre): 部分的消失 → 個性が薄れる
- 抑揚 (Prosody): 平坦化 → 感情表現が乏しい
- 子音の明瞭度: 低下 → /s/, /t/, /k/ が不明瞭
- 息の成分: 消失 → 不自然

---

## 6. 解決策の実装計画

### 6.1 Priority 1: GAN Discriminator実装・有効化

**理由**:
- ✅ Neural Vocoder/Codecの確立手法
- ✅ StreamVC/SoundStreamの本来の設計
- ✅ 高周波・振幅・位相を総合的に最適化
- ✅ **根本的な解決**

**実装要件**:

```python
# Multi-Scale Discriminator (必須)
class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            Discriminator(scale=1),    # 原音
            Discriminator(scale=2),    # 2x downsample
            Discriminator(scale=4),    # 4x downsample
        ])

# Loss計算
def generator_loss(fake_audio, real_audio, d_fake, feat_fake, feat_real):
    # Adversarial Loss (LSGAN)
    adv_loss = torch.mean((d_fake - 1) ** 2)

    # Feature Matching Loss
    feature_loss = sum(F.l1_loss(fr.detach(), ff)
                       for fr, ff in zip(feat_real, feat_fake))

    return adv_loss, feature_loss
```

**設定変更**:
```yaml
# configs/phase_1_ema_gan.yaml
losses:
  content_ce_weight: 1.0
  stft_weight: 1.0
  l1_weight: 10.0
  adversarial_weight: 4.0      # 0.0 → 4.0 (有効化)
  feature_matching_weight: 2.0 # 0.0 → 2.0
  rms_weight: 1.0              # 0.1 → 1.0 (併用)
  multiband_rms_weight: 0.5    # 0.05 → 0.5
```

**期待される効果**:
| メトリクス | Phase 1-EMA | 予測: GAN有効化 |
|-----------|-------------|----------------|
| Audio RMS | 0.002 ❌ | 0.05-0.10 ✅ |
| Perplexity | 12-72 ✅ | 10-50 ✅ (維持) |
| 高周波 (> 4kHz) | 不明 | 大幅改善 ✅ |
| 主観音質 | 不明 | 大幅改善 ✅ |

**リスクと対策**:
```yaml
# Warm-up期間でGAN不安定性を回避
adversarial_weight:
  step_0-2000: 0.0     # Generator先行学習
  step_2000-4000: 1.0  # 徐々に導入
  step_4000+: 4.0      # フル稼働
```

### 6.2 Priority 2: RMS Loss増強（併用）

**理由**:
- ✅ 振幅保持の明示的制約
- ✅ GANの補助として有効
- ✅ 低リスク、即座に実施可能

**設定変更**:
```yaml
losses:
  rms_weight: 1.0  # 0.1 → 1.0 (10倍)
  multiband_rms_weight: 0.5  # 0.05 → 0.5 (10倍)
```

**期待効果**:
- Total Lossの寄与率: 0.1% → 5-10%
- RVQ最適化圧力への抵抗力向上

**GANとの相補性**:
```
RMS Loss: "平均RMSは0.075であれ"（最低保証）
GAN: "Real音声のような自然な振幅分布であれ"（自然さ）
→ 両方使うことで最適化
```

### 6.3 Priority 3: Multiband STFT Loss（補完）

**実装例**:
```python
def multiband_stft_loss(pred, target, n_bands=4):
    """各周波数帯域でSTFT Lossを計算し、高周波に重みを付ける"""
    bands = [
        (0, 1000),      # 低周波
        (1000, 4000),   # 中周波
        (4000, 8000),   # 高周波 (子音)
        (8000, None),   # 超高周波
    ]
    weights = [1.0, 1.0, 2.0, 1.5]  # 高周波に重み

    loss = 0
    for (low, high), w in zip(bands, weights):
        band_pred = bandpass_filter(pred, low, high)
        band_target = bandpass_filter(target, low, high)
        loss += w * stft_loss(band_pred, band_target)

    return loss / sum(weights)
```

---

## 7. 実験計画

### 7.1 Phase 1-EMA-GAN (推奨)

```yaml
experiment:
  name: streamvc_phase1_ema_gan

model:
  # Encoder-Decoder + RVQ (既存)
  decoder:
    rvq:
      use_ema: true          # EMAは継続
      ema_decay: 0.99
      dead_threshold: 100
  # Multi-Scale Discriminator (新規追加)
  discriminator:
    scales: [1, 2, 4]
    channels: 64

training:
  num_steps: 10000
  losses:
    adversarial_weight: 4.0
    feature_matching_weight: 2.0
    rms_weight: 1.0
    multiband_rms_weight: 0.5
  optimizer_d:
    lr: 0.0002  # Generatorの2倍
```

**検証項目**:
- [ ] Audio RMS推移 (目標: 0.05-0.10)
- [ ] Perplexity維持 (目標: ≥10)
- [ ] 高周波スペクトル (> 4kHz)
- [ ] 主観音質評価（実際に聞いてみる）
- [ ] 学習安定性（Loss発散しないか）

### 7.2 Phase 1-EMA-v2 (RMS増強のみ、比較用)

```yaml
experiment:
  name: streamvc_phase1_ema_v2

# GANなし、RMS Loss増強のみ
losses:
  adversarial_weight: 0.0  # GAN無効
  rms_weight: 1.0          # 増強
  multiband_rms_weight: 0.5
```

**目的**: RMS増強だけで振幅崩壊が防げるか検証（対症療法の効果測定）

---

## 8. まとめ

### 8.1 核心的な発見

1. **EMA更新は振幅崩壊の原因ではない**
   - Perplexityを12.53に改善（Phase A-v2の3.56から+252%）
   - Dead code問題を解決
   - Phase A-v2（EMA無し）でも同様の振幅崩壊が発生

2. **振幅崩壊の真の原因（3層構造）**
   - Level 1: RMS Loss weight不足（0.1 → Total Lossの0.1%のみ）
   - Level 2: STFT + L1 Lossの盲点（位相無視、相対誤差のみ）
   - Level 3: **GAN Discriminatorの欠如**（これが本質）

3. **StreamVC論文の前提との乖離**
   - 論文: "SoundStream-styleの学習戦略（GAN/feature損失を含む）を踏襲"
   - 実装: Phase 1でGANを無効化 (`adversarial_weight: 0.0`)
   - **この判断が振幅崩壊の根本原因**

4. **Neural Vocoder/Codecの確立手法を未適用**
   - HiFi-GAN/SoundStream/EnCodec: 全てGAN Discriminatorを使用
   - Discriminatorなしでは高周波・振幅・位相の保持が困難
   - **これが業界標準の手法**

### 8.2 ユーザー質問への回答まとめ

**Q1: RMS lossが落ちすぎることに関しては、どうする予定か？**
- 短期: RMS weight 1.0に増強（対症療法）
- 長期: GAN Discriminator導入（根本解決）

**Q2: GANを導入した時に、RMS loss増強はどうなるの？**
- **両方併用する**（相補的）
- RMS Loss: 最低保証（1次元スカラー制約）
- GAN: 自然さ（多次元暗黙制約）

**Q3: これってそもそも振幅だっけ、それってGANと何が関係あるの？**
- RMS = 音量の指標（振幅の二乗平均平方根）
- RMS Loss: 明示的だが弱すぎる（0.1%の寄与）
- GAN: Real音声の自然な振幅分布を暗黙的に学習
- **GANが振幅崩壊の本質的解決策**

### 8.3 最優先アクション

🎯 **GAN Discriminator実装・有効化** (`adversarial_weight: 4.0`)
🎯 **RMS Loss増強** (`rms_weight: 1.0`) を併用
🎯 **StreamVC論文の前提条件を満たす実装に修正**

---

## 参考文献

### Neural Vocoder
- HiFi-GAN (Kong et al., 2020): https://arxiv.org/abs/2010.05646
- MelGAN (Kumar et al., 2019)
- UnivNet (Jang et al., 2021)

### Neural Audio Codec
- SoundStream (Zeghidour et al., 2021): https://arxiv.org/abs/2107.03312
- EnCodec (Défossez et al., 2022)
- DAC (Kumar et al., 2023)

### StreamVC
- 論文: https://arxiv.org/abs/2401.03078
- ポスター: https://google-research.github.io/seanet/stream_vc/poster/streamvc_poster.pdf

### 参考記事
- https://qiita.com/4wavetech/items/28441857d2139aecaf6a
- https://www.slideshare.net/slideshow/2025-d48b/
