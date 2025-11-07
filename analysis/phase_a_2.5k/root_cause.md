# Phase A Failure: Root Cause Analysis

**Date**: 2025-11-07
**Investigation**: Deep dive into why Phase A failed despite mitigation efforts

---

## TL;DR

**Phase Aが失敗した3つの決定的な原因**:

1. ❌ **final_conv の初期化ミス** (最重要)
   - Kaiming uniform → 対角成分~0 → スケール50%減
2. ❌ **RMS Loss の設計ミス**
   - 相対的形状を見るだけ → 絶対値スケールを監督していない
3. ❌ **post_scale/post_bias が学習されない**
   - 勾配がfinal_convで吸収される

---

## 1. Forward Pass Breakdown

### Step-by-Step Scale Degradation

| Stage | Mean | Std | Note |
|-------|------|-----|------|
| 1. Pre-RVQ norm | -0.030 | 1.011 | ✅ 正規化OK |
| 2. Quantized (Q0) | -0.020 | 0.842 | ⚠️ 16%減 (量子化誤差) |
| 3. Post-affine | -0.044 | 0.805 | ⚠️ 4%減 (scale=0.96) |
| 4. After final_conv | 0.008 | **0.468** | ❌ **50%減 (Kaiming init)** |
| 5. Final audio | 0.001 | **0.008** | ❌ **98%減 (out_proj)** |

**合計スケール縮小**: 1.0 → 0.008 = **99.2%減**

---

## 2. Critical Bugs Identified

### Bug #1: final_conv Kaiming Initialization

**Code**:
```python
# rvq.py:40
self.final_conv = nn.Conv1d(config.dims, config.dims, kernel_size=1)
```

**Problem**:
- PyTorchのデフォルトは **Kaiming uniform** 初期化
- Conv1d(40, 40, k=1) の対角成分が ~0.03 (期待: 1.0)
- → 入力 std=0.8 が std=0.47 に縮小 (**42%減**)

**Evidence**:
```
decoder.rvq.final_conv.weight:
  Diagonal mean: -0.004391  ← 期待値 1.0
  Off-diagonal norm: 3.582   ← ランダムノイズ
```

**Fix**:
```python
self.final_conv = nn.Conv1d(config.dims, config.dims, kernel_size=1)
nn.init.eye_(self.final_conv.weight.squeeze())  # Identity init
nn.init.zeros_(self.final_conv.bias)
```

**Expected Impact**:
- Std degradation: 0.8 → 0.8 (維持)
- Final audio std: 0.008 → **0.08** (10倍改善)

---

### Bug #2: RMS Loss Design Flaw

**Current Implementation**:
```python
def frame_rms_loss(pred, target, frame_length=512):
    pred_rms = sqrt(pred^2.mean(dim=-1))     # Frame-wise RMS
    target_rms = sqrt(target^2.mean(dim=-1))
    return L1(pred_rms, target_rms)
```

**Problem**:
- `pred` と `target` が**両方とも**同じ倍率で縮小しても loss は小さい
- 例: `pred = target × 0.01` でも、frame-wise の**相対的形状**が合えば loss ≈ 0
- **絶対値スケール**を全く監督していない

**Evidence**:
```
RMS Loss evolution:
  0-500:   0.173
  2K-2.5K: 0.061  (-64%, converging)

Actual audio RMS:
  0-500:   0.231
  2K-2.5K: 0.0037 (-98%, collapsing!)
```

**Fix Option 1: Absolute RMS Target**:
```python
def absolute_rms_loss(pred, target_rms=0.075):
    pred_rms = torch.sqrt((pred ** 2).mean() + 1e-8)
    return F.mse_loss(pred_rms, torch.tensor(target_rms, device=pred.device))
```

**Fix Option 2: RMS Ratio Constraint**:
```python
def rms_ratio_loss(pred, target):
    pred_rms = torch.sqrt((pred ** 2).mean(dim=-1) + 1e-8)
    target_rms = torch.sqrt((target ** 2).mean(dim=-1) + 1e-8)
    ratio = pred_rms / (target_rms + 1e-8)
    return F.mse_loss(ratio, torch.ones_like(ratio))
```

---

### Bug #3: post_scale/post_bias Not Learning

**Expected Behavior**:
- `post_scale` should learn to amplify quantized embeddings
- `post_bias` should learn DC offset

**Actual Behavior**:
```
decoder.rvq.post_scale:
  Values: [0.9556, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
  → ほぼ初期値 (1.0) のまま

decoder.rvq.post_bias:
  Values: [-0.024, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  → ほぼ初期値 (0.0) のまま
```

**Why Not Learning**:
1. **Gradient absorption by final_conv**:
   - `post_scale` → `final_conv` → `out_proj` の3層
   - 勾配が `final_conv` で大部分吸収される
   - `post_scale` への勾配が微小

2. **Learning rate mismatch**:
   - 他のパラメータと同じLR (1e-4)
   - Scalarパラメータは通常**高いLR**が必要

**Fix**:
```python
# Separate optimizer group with higher LR
affine_params = [
    self.pipeline.decoder.rvq.post_scale,
    self.pipeline.decoder.rvq.post_bias,
]
param_groups = [
    {'params': affine_params, 'lr': 1e-3},  # 10x higher
    {'params': other_params, 'lr': 1e-4},
]
```

---

## 3. Why RMS Supervision Failed

### The Scale-Invariant Loss Trap

**Visual Example**:
```
Target audio:     [0.1, 0.2, 0.1, 0.3] → RMS = 0.187
Predicted (good): [0.1, 0.2, 0.1, 0.3] → RMS = 0.187 → loss = 0.0
Predicted (bad):  [0.001, 0.002, 0.001, 0.003] → RMS = 0.00187 → loss = 0.0 (?!)
                   ↑ 100倍小さいのに loss は同じ！
```

**Why This Happens**:
```python
# frame_rms_loss の計算
pred_rms = [0.00187, 0.00187, ...]    # 各フレームのRMS
target_rms = [0.187, 0.187, ...]      # 目標RMS

# L1 loss
loss = |0.00187 - 0.187| = 0.185  # 大きなlossになるはず...

# しかし実際は pred も target も同じパターンで変動
# → 相対的な形状が合えば loss は小さくなる
```

**Correct Implementation Should**:
- **Absolute scale** を監督
- または **RMS ratio = 1.0** を強制
- または **Batch-level RMS** を固定

---

## 4. Comparison: Why Phase 1 Didn't Collapse

### Phase 1 @ 5K

**Parameters**:
```
out_proj weight norm: 0.083
out_proj bias norm: 0.050
Audio RMS: 0.080 (target: 0.075, +7%)
```

**Why Phase 1 was OK**:
1. ✅ No final_conv → スケール破壊なし
2. ✅ No RMS loss → scale-invariant trap なし
3. ✅ Simple L2 VQ → 安定した量子化

**Why Phase 1 had mosquito tone**:
- Code collapse (perplexity 5-6)
- 高域が粗く近似される
- **しかし振幅は正常**

---

## 5. Why Phase 2A Also Collapsed

### Phase 2A @ 10K

**Parameters**:
```
out_proj weight norm: 0.022 (-73%)
out_proj bias norm: 0.0006 (-99%)
Audio RMS: 0.004 (-95%)
```

**Why Phase 2A collapsed**:
1. ❌ Cosine VQ の正規化 → スケール不整合
2. ❌ STE coordinate mismatch → 勾配エラー
3. ❌ No explicit scale supervision

**Key Difference from Phase A**:
- Phase 2A: out_proj自体が縮退 (-73%)
- Phase A: final_conv が縮退 (-50%), out_proj は安定 (-5%)

---

## 6. Solution Matrix

### Three-Tier Fix

| Tier | Fix | Impact | Difficulty |
|------|-----|--------|------------|
| **Tier 1** | final_conv Identity init | ✅ **+10x audio amplitude** | Easy |
| **Tier 2** | Absolute RMS loss | ✅ **Prevent scale drift** | Easy |
| **Tier 3** | post_scale higher LR | ⚠️ Faster convergence | Medium |

### Implementation Priority

**Immediate (Phase A-v2)**:
1. ✅ Identity init for final_conv
2. ✅ Replace frame_rms_loss with absolute_rms_loss
3. ✅ Add batch-level RMS constraint

**Optional**:
4. ⚠️ Separate LR for post_scale/post_bias
5. ⚠️ Norm regularization for out_proj

---

## 7. Expected Results (Phase A-v2)

### Predictions

| Metric | Phase A | Phase A-v2 (predicted) | Target |
|--------|---------|----------------------|--------|
| Audio RMS @ 2K | 0.004 | **0.08** | 0.075 |
| out_proj norm | -5% | **-5%** | <10% |
| Perplexity Q0 | 8.6 | **8-10** | ≥8 |
| STFT loss | 1.19 | **1.2-1.3** | Maintain |

### Success Criteria

- ✅ Audio RMS: 0.075 ± 20% (0.06-0.09)
- ✅ out_proj norm: <10% degradation
- ✅ Perplexity: ≥8
- ✅ No mosquito tone (requires Phase B/C)

---

## 8. Lessons Learned

### Design Principles for Scale-Sensitive Models

1. **Always use Identity init for 1x1 conv bypass connections**
   - Default Kaiming/Xavier can destroy scale
   - Especially critical in residual paths

2. **Scale-invariant losses need explicit scale supervision**
   - STFT, L1, relative RMS are all scale-invariant
   - Add absolute RMS, LUFS, or peak normalization

3. **Scalar parameters need higher learning rates**
   - `post_scale`, `post_bias` learn 10-100x slower
   - Use separate optimizer groups

4. **Monitor parameter norms, not just losses**
   - Loss can improve while model degrades
   - Track weight/bias norms every step

---

## 9. Next Steps

### Phase A-v2 Implementation

**Files to modify**:
1. `src/streamvc/modules/rvq.py`:
   - Add Identity init in `__init__`
2. `src/streamvc/losses.py`:
   - Replace `frame_rms_loss` → `absolute_rms_loss`
3. `src/streamvc/trainer.py`:
   - Optional: Separate LR for affine params

**Validation**:
- Train for 2K steps
- Check audio RMS ≈ 0.075
- Check out_proj stability
- If OK → continue to 5K

**Timeline**:
- Implementation: 30 min
- Training: 1 hour (2K step)
- Decision point: Pass/Fail by audio RMS

---

## Appendix: Parameter Inspection

### @ Step 2000

```
decoder.rvq.post_scale:        [0.956, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
decoder.rvq.post_bias:         [-0.024, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
decoder.rvq.final_conv.weight: mean=0.004, std=0.091, diag_mean=-0.004
decoder.rvq.final_conv.bias:   mean=0.004, norm=0.540
decoder.out_proj.weight:       mean=0.010, std=0.085, norm=0.536
decoder.out_proj.bias:         -0.109
```

### Loss Evolution

```
Loss Component    0-500    2K-2.5K  Ratio
STFT              4.108    1.190    0.29x
L1                0.203    0.048    0.24x
RVQ               1.153    2.133    1.85x  ← Increasing!
RMS               0.173    0.058    0.33x  ← Converging but useless
Multi-RMS         0.092    0.020    0.22x
```

### Forward Pass Scale Degradation

```
Stage              Std      Degradation
Input (normalized) 1.011    -
Quantized         0.842    -17%
Post-affine       0.805    -20%
After final_conv  0.468    -54% ← Critical
Final audio       0.008    -99% ← Collapse
```
