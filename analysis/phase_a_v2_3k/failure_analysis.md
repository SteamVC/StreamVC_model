# Phase A-v2 @ 3K: 振幅崩壊が再発

**日付**: 2025-11-07
**ステータス**: ❌ **FAILED** - Phase Aと同じ振幅崩壊が発生

---

## TL;DR

**Phase A-v2も失敗した根本原因**:

1. ✅ **final_conv Identity初期化は成功** - 対角成分 0.98（正常）
2. ✅ **Absolute RMS Lossは実装された** - Loss自体は収束
3. ❌ **Pre-RVQ正規化がスケール情報を破壊** - これが真の原因
4. ❌ **out_projがスケール再構築に失敗** - 18%劣化

**結論**: Identity初期化とRMS Loss修正だけでは不十分。**Pre-RVQ正規化そのものが問題**。

---

## 1. Phase A-v2 @ 3K Step 結果サマリ

### メトリクス比較

| メトリクス | Phase A @ 2.5K | Phase A-v2 @ 3K | 目標値 | 判定 |
|-----------|----------------|-----------------|--------|------|
| **Audio RMS** | 0.0037 | **0.0044** | 0.075 | ❌ -94% |
| **Perplexity Q0** | 8.6 | **3.56** | ≥8 | ❌ -56% |
| **out_proj norm** | -5% | **-18%** | <10% | ❌ 悪化 |
| **final_conv norm** | 0.47 | **6.26** | ~6.3 | ✅ 正常 |
| **STFT Loss** | 1.19 | **1.20** | - | ✅ 同等 |
| **L1 Loss** | 0.048 | **0.054** | - | ⚠️ 微増 |

### トレンド分析（Step 100 → 3000）

```
Audio RMS:         0.523 → 0.004  (-99.2%, 崩壊)
Perplexity Q0:     28.1 → 3.56   (-87.3%, コード崩壊)
out_proj norm:     0.565 → 0.463 (-18.1%, 劣化)
final_conv norm:   6.32 → 6.26   (-1.0%, 安定)
RMS Loss:          0.195 → 0.008 (-96.1%, 収束)
```

**重要な観察**:
- **RMS Lossは正常に収束**しているが、**実際の音声振幅は崩壊**
- final_convは安定しているが、out_projが劣化
- Phase Aと本質的に同じ結果

---

## 2. 適用した修正の検証

### 修正1: final_conv Identity初期化

**コード**:
```python
# src/streamvc/modules/rvq.py:42
self.final_conv = nn.Conv1d(config.dims, config.dims, kernel_size=1)
nn.init.eye_(self.final_conv.weight.squeeze())
nn.init.zeros_(self.final_conv.bias)
```

**検証結果**:
```
@ Step 3000:
  final_conv.weight diagonal mean: 0.976 (期待: ~1.0)
  final_conv.weight norm: 6.26 (初期: ~6.3)
  final_conv.bias norm: 0.22
```

✅ **成功**: Identity初期化は正しく適用され、学習中も安定

### 修正2: Absolute RMS Loss

**コード**:
```python
# src/streamvc/losses.py:95-99
pred_rms = torch.sqrt((prediction ** 2).mean() + 1e-8)
target_rms = torch.sqrt((target ** 2).mean() + 1e-8)
return F.mse_loss(pred_rms, target_rms)
```

**検証結果**:
```
RMS Loss推移:
  0-500:   0.173
  2.5K-3K: 0.008 (-95%, 収束)

実際のAudio RMS:
  0-500:   0.525
  2.5K-3K: 0.004 (-99%, 崩壊!)
```

⚠️ **Loss収束 ≠ 振幅維持**: Lossは小さくなったが、振幅は崩壊

---

## 3. 真の原因: Pre-RVQ正規化

### 問題のコード

**decoder.py:126-127**:
```python
# Manual normalization: mean=0, std=1
x_for_rvq = x_for_rvq - x_for_rvq.mean(dim=-1, keepdim=True)
x_for_rvq = x_for_rvq / (x_for_rvq.std(dim=-1, keepdim=True) + 1e-5)
```

### なぜこれが問題なのか

**スケール情報の破壊**:
```
Before normalization:
  x_for_rvq std: 3.5 (チャネル依存、スケール情報あり)

After normalization:
  x_for_rvq std: 1.0 (全チャネル強制的に1.0、スケール情報消失)
```

**フォワードパスの流れ**:
1. `post(x)` → std ≈ 3.5 (スケール情報あり)
2. `pre_rvq_conv(x)` → std ≈ 3.5 (維持)
3. **正規化** → std = 1.0 (**スケール情報破壊**)
4. `rvq(x_normalized)` → std ≈ 0.98 (正規化空間)
5. `final_conv(quantized)` → std ≈ 0.98 (Identity保存)
6. **out_proj(quantized)** → std ≈ 0.004 (**スケール再構築失敗**)

### なぜout_projが失敗するのか

**スケール再構築の難しさ**:
- 正規化で失われたスケール情報を**out_projが暗黙的に記憶**する必要がある
- しかし、out_projは1x1 Conv（単純な線形変換）
- 学習中に他のLoss（STFT, L1）が支配的 → スケール情報が学習されない
- 結果: out_proj norm が徐々に縮小（0.565 → 0.463, -18%）

**なぜRMS Lossが効かないのか**:
```python
# RMS Loss計算
pred_rms = sqrt(pred^2.mean())     # pred = out_proj(quantized)
target_rms = sqrt(target^2.mean()) # target ≈ 0.075

# 問題点
# 1. RMS Lossは1スカラー値しか監督しない
# 2. STFT/L1 Lossは何百次元もあり、勾配が支配的
# 3. out_projは「STFT/L1を最小化しつつRMSも満たす」を学習
# 4. → STFT/L1が優先され、RMSは犠牲になる
```

---

## 4. パラメータ詳細分析

### @ Step 3000

```
decoder.rvq.final_conv:
  weight diagonal mean: 0.9760 (✅ Identity維持)
  weight norm:          6.2631
  bias norm:            0.2155

decoder.rvq.post_scale:
  [0.908, 0.973, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
  → Q0, Q1は学習中（0.91, 0.97）

decoder.rvq.post_bias:
  [0.028, -0.003, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  → わずかに学習

decoder.out_proj:
  weight mean:  0.006
  weight std:   0.074
  weight norm:  0.463 (初期: ~0.55, -18%)
  bias:        -0.099

decoder.pre_rvq_conv:
  weight norm:  3.713
```

**問題点**:
- out_proj normが-18%劣化 → スケール縮小
- 正規化がスケール情報を破壊しているため、out_projが補償しきれない

---

## 5. なぜPhase 1は成功したのか

### Phase 1 @ 5K パラメータ

```
Audio RMS:         0.080 (目標: 0.075, +7% ✅)
out_proj norm:     0.083
Perplexity:        5-6 (低いがモスキートーン)
```

### Phase 1との違い

| 要素 | Phase 1 | Phase A-v2 | 影響 |
|------|---------|------------|------|
| Pre-RVQ正規化 | ✅ **あり** | ✅ あり | 同じ |
| final_conv | ❌ **なし** | ✅ あり | - |
| post_scale/bias | ❌ **なし** | ✅ あり | - |
| RMS Loss | ❌ **なし** | ✅ あり | - |
| STE consistency | ❌ **なし** | ✅ あり | - |

**なぜPhase 1は振幅維持できたのか**:

**仮説1**: シンプルなアーキテクチャ
- Phase 1: `rvq → out_proj` （2層）
- Phase A-v2: `rvq → post_affine → final_conv → out_proj` （4層）
- → 層が少ない = 勾配が直接out_projに届く

**仮説2**: 初期化の運
- Phase 1のout_proj初期化がたまたま良かった
- Phase A-v2はfinal_convの追加で勾配フローが変化

**仮説3**: RVQ Loss配分
- Phase 1: RVQ Lossが相対的に大きい → 量子化誤差が監督される
- Phase A-v2: RMS/Multi-band RMS Lossが追加 → RVQ Lossの影響が希薄化

---

## 6. なぜ1K stepでは成功に見えたのか

### 1K step時点の数値

```
Audio RMS:         0.135 (目標: 0.075の1.8倍)
final_conv norm:   6.29 (Identity維持)
Perplexity Q0:     11.9
out_proj norm:     0.52 (-8%)
```

**この時点では成功に見えた理由**:
1. Audio RMSがまだ高い（0.135）
2. out_proj劣化が軽微（-8%）
3. Perplexityも健全（11.9）

**しかし内部では崩壊が進行中**:
```
Audio RMS推移:
  100:   0.523
  500:   0.363
  1000:  0.177  ← この時点の報告
  1500:  0.089
  2000:  0.032
  3000:  0.004  ← 崩壊完了
```

→ **指数関数的な減衰**が発生していた

---

## 7. 修正方針の検討

### オプション1: Pre-RVQ正規化を削除

**変更**:
```python
# decoder.py:122-127 を削除
# x_for_rvq = self.pre_rvq_conv(x)  # (B, C, T)
# ↓
# x_for_rvq = x_for_rvq - x_for_rvq.mean(dim=-1, keepdim=True)
# x_for_rvq = x_for_rvq / (x_for_rvq.std(dim=-1, keepdim=True) + 1e-5)
# ↓
x_for_rvq = self.pre_rvq_conv(x)  # 正規化なし
```

**メリット**:
- スケール情報が保存される
- out_projがスケールを記憶する必要なし

**デメリット**:
- RVQの入力分布が不安定になる可能性
- Codebookの初期化が適切でない可能性

**リスク**: ⚠️ 中 - RVQ学習が不安定化する可能性

---

### オプション2: スケール情報を明示的に伝播

**変更**:
```python
# decoder.py forward()内
pre_norm_std = x_for_rvq.std(dim=-1, keepdim=True)  # スケール保存
x_for_rvq = x_for_rvq / (pre_norm_std + 1e-5)      # 正規化

quantized, rvq_loss, codes = self.rvq(x_for_rvq)

# スケール復元
quantized = quantized * pre_norm_std.squeeze(-1).unsqueeze(1)  # (B, T, C)

audio = self.out_proj(quantized.transpose(1, 2)).squeeze(1)
```

**メリット**:
- RVQは正規化空間で動作（安定）
- スケール情報は明示的に復元

**デメリット**:
- 実装が複雑
- スケール復元時の次元操作が必要

**リスク**: ⚠️ 低 - 理論的に正しい

---

### オプション3: LayerNorm + Learnable Affine

**変更**:
```python
# decoder.py __init__()
self.pre_rvq_norm = nn.LayerNorm(config.channels)  # Learnable scale/bias

# forward()
x_for_rvq = self.pre_rvq_conv(x).transpose(1, 2)  # (B, T, C)
x_for_rvq = self.pre_rvq_norm(x_for_rvq)          # Learnable affine
quantized, rvq_loss, codes = self.rvq(x_for_rvq)
```

**メリット**:
- LayerNormのaffineパラメータがスケール学習
- 標準的なアプローチ

**デメリット**:
- LayerNorm自体は正規化するので、スケール情報は依然破壊される
- affineパラメータがスケールを記憶する必要がある

**リスク**: ⚠️ 中 - 現状と本質的に同じ問題

---

### オプション4: RVQを非正規化空間で動作させる

**変更**:
```python
# RVQのCodebookを大きなスケールで初期化
# rvq.py __init__()
self.codebooks = nn.ParameterList([
    nn.Parameter(torch.randn(config.codebook_size, config.dims) * 3.0)  # スケール3.0
    for _ in range(config.num_quantizers)
])

# decoder.py: 正規化なし
x_for_rvq = self.pre_rvq_conv(x).transpose(1, 2)
quantized, rvq_loss, codes = self.rvq(x_for_rvq)  # 非正規化空間
```

**メリット**:
- スケール情報完全保存
- シンプル

**デメリット**:
- Codebookスケールの調整が必要
- RVQ学習が不安定になる可能性

**リスク**: ⚠️ 高 - Codebook初期化が難しい

---

## 8. 推奨アプローチ

### Phase A-v3: オプション2（スケール明示伝播）

**理由**:
1. ✅ スケール情報を破壊しない
2. ✅ RVQは正規化空間で安定動作
3. ✅ 理論的に正しい
4. ✅ 実装が明確

**実装手順**:
1. `decoder.py`でpre_norm_stdを保存
2. RVQ出力にスケールを乗算
3. out_projは通常通り動作

**期待される結果**:
- Audio RMS: 0.075 ± 10%
- out_proj安定（劣化<5%）
- Perplexity: ≥8

---

## 9. Lessons Learned

### 正規化とスケール保存の両立

**問題**:
- 正規化 = スケール情報破壊
- VQは正規化空間で安定
- 出力は絶対スケールが必要

**解決策**:
- **正規化前にスケールを保存**
- **正規化空間でVQ**
- **明示的にスケール復元**

### Lossの優先度

**問題**:
- RMS Loss（1次元）vs STFT/L1 Loss（数百次元）
- 勾配の規模が圧倒的に違う
- RMSが無視される

**解決策**:
- スケール情報を**アーキテクチャで保証**
- Lossに頼らない

---

## 10. 次のステップ

### Phase A-v3実装

**ファイル**:
1. `src/streamvc/modules/decoder.py`:
   - スケール明示伝播の実装

**検証**:
- 2K stepで振幅確認
- Audio RMS ≈ 0.075なら5K継続

**タイムライン**:
- 実装: 15分
- 学習: 1時間（2K step）
- 判定: Pass/Fail

---

## Appendix: 詳細メトリクス

### Loss推移（Last 30 steps: 100-3000）

```
train/audio_rms_mean:
  Range: 0.00165 - 0.52318
  Latest: 0.00436
  Trend: 0.52318 → 0.00436 (-99.2%)

train/rvq_perplexity_q0:
  Range: 3.43 - 37.09
  Latest: 3.56
  Trend: 28.06 → 3.56 (-87.3%)

train/out_proj_weight_norm:
  Range: 0.463 - 0.565
  Latest: 0.463
  Trend: 0.565 → 0.463 (-18.1%)

train/final_conv_norm:
  Range: 6.26 - 6.32
  Latest: 6.26
  Trend: 6.32 → 6.26 (-1.0%)

train/loss_stft:
  Range: 1.15 - 8.53
  Latest: 1.20
  Trend: 8.46 → 1.20 (-85.8%)

train/loss_rms:
  Range: 0.00005 - 0.19504
  Latest: 0.00759
  Trend: 0.19504 → 0.00759 (-96.1%)
```

### パラメータ比較

| パラメータ | Phase 1 @ 5K | Phase A @ 2.5K | Phase A-v2 @ 3K |
|-----------|--------------|----------------|-----------------|
| Audio RMS | 0.080 ✅ | 0.0037 ❌ | 0.0044 ❌ |
| out_proj norm | 0.083 | 0.536 | 0.463 |
| Perplexity | 5-6 | 8.6 | 3.56 |
| final_conv | - | 0.47 (Bug) | 6.26 (✅) |

---

**結論**: Phase A-v2の修正は正しかったが、**Pre-RVQ正規化という根本問題**を見落としていた。Phase A-v3でスケール明示伝播を実装する必要がある。
