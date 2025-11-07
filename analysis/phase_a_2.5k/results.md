# Phase A @ 2.5K Step - Analysis Report

**Date**: 2025-11-07
**Checkpoint**: runs/streamvc_phase_a_scale_fix/checkpoints/step_2000.pt (interpolated to 2.5K)

## Executive Summary

🚨 **CRITICAL FAILURE**: Phase A exhibits the **same amplitude collapse** as Phase 2A despite all mitigation efforts.

- ✅ out_proj norm: Stable (-5.6%, target <10%)
- ✅ Perplexity Q0: 8.55 (target ≥8)
- ❌ **Audio RMS: 0.0037 (95% below target 0.075)**
- ❌ RMS Loss降下 (-64%) but 実際の振幅は崩壊

## Metrics Comparison

### Loss Trends (0→2.5K step)

| Metric | 0-500 | 1K-1.5K | 2K-2.5K | Change |
|--------|-------|---------|---------|--------|
| **Total Loss** | 12.03 | 7.62 | 8.33 | -30.8% |
| STFT Loss | 4.11 | 1.27 | 1.20 | -70.8% |
| L1 Loss | 0.203 | 0.057 | 0.052 | -74.5% |
| RVQ Loss | 1.15 | 1.24 | 2.16 | **+87.5%** ⚠️ |
| **RMS Loss** | 0.173 | 0.056 | 0.061 | -64.4% |
| Multi-band RMS | 0.092 | 0.023 | 0.021 | -76.7% |

### RVQ Diagnostics

| Metric | 0-500 | 1K-1.5K | 2K-2.5K | Target | Status |
|--------|-------|---------|---------|--------|--------|
| Perplexity Q0 | 25.6 | 11.1 | 8.6 | ≥8 | ✓ Good |
| Perplexity Q1 | - | - | 16.2 | ≥8 | ✓ Good |

### Critical Parameters

| Parameter | Initial | @ 2.5K | Change | Target | Status |
|-----------|---------|--------|--------|--------|--------|
| **out_proj W norm** | 0.564 | 0.536 | -5.0% | <10% | ✓ **Stable** |
| out_proj B norm | 0.135 | 0.109 | -19.3% | <10% | ⚠️ Degrading |
| final_conv norm | 3.659 | 3.630 | -0.8% | Stable | ✓ Stable |
| **Audio RMS** | 0.231 | **0.0037** | **-97.9%** | 0.075±20% | ❌ **CRITICAL** |

## Problem Analysis

### 1. RMS Loss is Working BUT Ineffective

**Paradox**:
- RMS Loss: 0.173 → 0.061 (-64%, converging)
- Multi-band RMS: 0.092 → 0.021 (-77%, converging)
- **Actual Audio RMS: 0.231 → 0.0037 (-98%, collapsing)**

**Interpretation**:
- Loss関数は「小さな振幅でもRMSを合わせられる」と学習
- Target RMSが0.075でも、pred/targetが両方ゼロに近づけば loss=0
- **Scale-invariant lossの根本的問題**

### 2. RVQ Loss Increase (+87.5%)

- Phase 2Aと同じパターン: RVQ loss↑ = Code collapse進行
- Perplexityは8.6で良好なのに、なぜRVQ lossが増加？
- **仮説**: Encoder出力とCodebookの座標が乖離
  - Pre-RVQ正規化でstd=1に固定
  - しかしCodebookは学習中に変化
  - → Commitment loss増加

### 3. out_proj vs final_conv

**Good news**:
- out_proj W: -5.0% (Phase 2A: -73%)
- final_conv: -0.8% (ほぼ安定)

**Bad news**:
- out_proj B: -19.3% (weight_decay=0でも減少)
- Biasだけが減少 → 全体のDCオフセットが減少 → 振幅崩壊

### 4. STE Fix is Not Fixing

実装した STE consistency fix:
```python
# Phase A implementation
embeds_st = residual + (embeds - residual).detach()
embeds_scaled = embeds_st * post_scale + post_bias
```

**しかし**:
- 依然として振幅崩壊が発生
- `post_scale`/`post_bias`が適切に機能していない可能性
- または `final_conv` の後処理が不足

## Root Cause Hypothesis

### Why RMS Supervision Failed

**Target matching の問題**:
```python
# frame_rms_loss の実装
pred_rms = sqrt(pred^2.mean())   # Pred RMSを計算
target_rms = sqrt(target^2.mean()) # Target RMSを計算
loss = L1(pred_rms, target_rms)    # L1 loss
```

**問題点**:
- Predが全体的にスケールダウンしても、**相対的な形状が保たれればlossは下がる**
- 例: pred = target × 0.01 でも、frame-wise RMSの**比率**が合えばloss小

**必要な修正**:
- 絶対的なRMSターゲットを設定
- または、RMS **ratio** を1.0に固定する制約

### Why out_proj Bias Collapsed

**weight_decay=0でも減少**:
- Gradient descent自体がbiasを縮小方向に誘導
- 理由: 小さな出力 = 小さなloss (STFT/L1が scale-invariant)
- **Explicit regularization** が必要（減少を積極的に防ぐ）

## Recommendations

### Immediate Action (緊急修正)

**Option 1: RMS Loss を絶対値ターゲットに変更**
```python
def absolute_rms_loss(pred, target_rms=0.075):
    pred_rms = torch.sqrt((pred ** 2).mean() + 1e-8)
    return F.mse_loss(pred_rms, torch.tensor(target_rms, device=pred.device))
```

**Option 2: Scale Anchor Loss (スケール固定)**
```python
def scale_anchor_loss(pred, target):
    # Force pred and target to have same RMS
    pred_rms = torch.sqrt((pred ** 2).mean() + 1e-8)
    target_rms = torch.sqrt((target ** 2).mean() + 1e-8)
    scale_ratio = target_rms / (pred_rms + 1e-8)

    # Penalize deviation from scale_ratio=1.0
    return F.mse_loss(scale_ratio, torch.ones_like(scale_ratio))
```

**Option 3: out_proj/final_conv に Norm Regularization**
```python
# In train_step
out_proj_norm_target = 0.5  # Initial norm
norm_reg = F.mse_loss(
    out_proj_weight_norm,
    torch.tensor(out_proj_norm_target, device=device)
)
total_loss += 0.01 * norm_reg
```

### Next Steps

1. **Kill current training** (振幅崩壊が進行中)
2. **Phase A-v2 実装**:
   - Absolute RMS target loss
   - Scale anchor loss
   - Norm regularization
3. **2K stepで早期検証**
4. 成功なら5K→10Kへ継続

## Phase Comparison

| Phase | @ 2.5K RMS | @ 2.5K out_proj | @ 2.5K Perplexity | Verdict |
|-------|-----------|-----------------|-------------------|---------|
| Phase 1 | ~0.080 | 0.083 | 5-6 | モスキートーン、振幅OK |
| Phase 2A | **0.004** | **0.022 (-73%)** | 16→8 | 振幅崩壊、perp一時改善 |
| **Phase A** | **0.0037** | 0.536 (-5%) | 8.6 | **振幅崩壊、out_proj安定は不十分** |

## Conclusion

**Phase A の評価: FAILED**

- ✅ STE fix: 実装済みだが効果なし
- ✅ out_proj weight: 安定 (-5%)
- ❌ **RMS supervision: 実装済みだがスケール崩壊を防げず**
- ❌ **Audio RMS: Phase 2Aと同レベルの崩壊 (98%減)**

**根本原因**:
- Scale-invariant loss (STFT, L1, RMS) の構造的問題
- "小さく出力すれば勝ち" の抜け道が依然として存在

**Next Action**:
- Phase A-v2 で絶対値ターゲット + Norm正則化を導入
- または Phase 1に戻ってGAN追加（スケール問題を判別器に任せる）
