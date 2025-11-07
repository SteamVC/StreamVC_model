# StreamVC RVQ Fix: Phase-by-Phase Plan

## 問題の診断結果

### 二段階崩壊メカニズム
1. **Code Collapse** (量子化レベル)
   - 少数コード化 → RVQ Loss↑ (0.76→1.22, +61%)
   - Perplexity↓ (Q0: 16→10, Q1: 16→8)
   - 相関係数: -0.56 (強い負相関)
   - 結果: 表現力不足、モスキートーン

2. **Output Scale Collapse** (出力レベル)
   - STE/正規化の座標ズレ
   - out_proj縮退 (weight -73%, bias -99%)
   - STFT/L1がスケール不変 → 小振幅でも"正解"
   - 結果: 音声振幅96%減 (std 0.21→0.004)

---

## Phase A: スケール崩壊の修正（最優先）

### 目標
- out_proj の縮退を防ぐ
- 振幅スケールを監督下に置く
- STE/正規化の座標整合

### 変更内容

#### A1. STE整合化 (rvq.py)
```python
# 変更前: quantized = x + (quantized - x).detach()
# 問題: x と quantized が異なる座標系

# 変更後: 同じ正規化空間で量子化、post-scaleで復元
class ResidualVectorQuantizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.codebooks = nn.ParameterList(...)
        # Post-quantization scale/bias (learnable)
        self.post_scale = nn.Parameter(torch.ones(config.num_quantizers))
        self.post_bias = nn.Parameter(torch.zeros(config.num_quantizers))

    def forward(self, x):
        # x already normalized in decoder (mean=0, std=1)
        residual = x
        quantized = torch.zeros_like(x)

        for q_idx, codebook in enumerate(active_codebooks):
            # L2 distance in normalized space
            indices = self._find_nearest(residual, codebook)
            embeds = F.embedding(indices, codebook)

            # Apply learnable post-scale/bias
            embeds = embeds * self.post_scale[q_idx] + self.post_bias[q_idx]

            quantized = quantized + embeds
            residual = residual - embeds
            commitment_loss += ...

        # STE in the SAME coordinate space
        quantized = x + (quantized - x).detach()
        return quantized, commitment_loss, codes
```

#### A2. RMS監督 (losses.py)
```python
def frame_rms_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Frame-wise RMS matching to prevent amplitude collapse."""
    rms_pred = torch.sqrt((pred**2).mean(dim=-1, keepdim=True) + 1e-8)
    rms_target = torch.sqrt((target**2).mean(dim=-1, keepdim=True) + 1e-8)
    return F.l1_loss(rms_pred, rms_target)

def multiband_rms_loss(pred: torch.Tensor, target: torch.Tensor,
                       n_bands: int = 4) -> torch.Tensor:
    """Multi-band RMS to prevent high-freq collapse."""
    import torchaudio.transforms as T

    # Mel-scale band split
    losses = []
    for i in range(n_bands):
        # Filter to i-th mel band
        # ... (省略: bandpass filtering)
        band_loss = frame_rms_loss(pred_band, target_band)
        losses.append(band_loss)

    return torch.stack(losses).mean()
```

#### A3. Loss統合 (trainer.py)
```python
# train_step() 内
rms = frame_rms_loss(generated, target_wave)
multiband_rms = multiband_rms_loss(generated, target_wave)

total = (
    self.loss_weights.content_ce * ce
    + self.loss_weights.l1 * l1
    + self.loss_weights.stft * stft
    + rvq
    + 0.1 * rms           # RMS監督
    + 0.05 * multiband_rms  # Multi-band RMS
)
```

#### A4. out_proj の weight decay = 0
```python
# _build_optimizer() 内
decoder_out_proj_params = [
    p for n, p in self.pipeline.named_parameters()
    if 'decoder.out_proj' in n
]
other_params = [
    p for n, p in self.pipeline.named_parameters()
    if 'decoder.out_proj' not in n
]

return torch.optim.AdamW([
    {'params': other_params, 'weight_decay': opt_cfg.get('weight_decay', 0.0)},
    {'params': decoder_out_proj_params, 'weight_decay': 0.0},  # No decay
], lr=opt_cfg['lr'], betas=...)
```

### 検証指標 (1K/2K/5K)
- ✅ out_proj weight norm: 安定 (減少率 <10%)
- ✅ 音声 RMS: target比 ±0.5 dB 以内
- ✅ High-freq RMS (>6kHz) vs Mid-freq (1-4kHz): 比率 ±3dB
- ✅ Perplexity Q0: ≥8 (Phase 2A @ 2K 並み)

### Config
- ファイル: `configs/phase_a_scale_fix.yaml`
- Base: Phase 1 (L2 VQ, no Cosine)
- 追加: RMS監督、out_proj保護
- Steps: 5K (早期検証)

---

## Phase B: EMA-VQ導入（コード多様性回復）

### 目標
- Code collapse防止
- Perplexity ≥10 維持
- Commitment loss依存を軽減

### 変更内容

#### B1. EMA-VQ実装 (rvq.py)
```python
class EMAVectorQuantizer(nn.Module):
    """VQ-VAE v2 style EMA-based VQ."""
    def __init__(self, config):
        super().__init__()
        self.codebook_size = config.codebook_size
        self.dims = config.dims

        # Codebook (not learnable via gradient)
        self.register_buffer('codebook', torch.randn(config.codebook_size, config.dims))
        self.register_buffer('cluster_size', torch.zeros(config.codebook_size))
        self.register_buffer('embed_avg', self.codebook.clone())

        self.decay = 0.99
        self.epsilon = 1e-5

    def forward(self, x):
        # Find nearest code
        distances = torch.cdist(x, self.codebook)
        indices = distances.argmin(dim=-1)
        quantized = F.embedding(indices, self.codebook)

        if self.training:
            # EMA update
            encodings = F.one_hot(indices, self.codebook_size).float()
            self.cluster_size.mul_(self.decay).add_(
                encodings.sum(0), alpha=1 - self.decay
            )
            embed_sum = encodings.transpose(0, 1) @ x.reshape(-1, self.dims)
            self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

            # Laplace smoothing
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.epsilon)
                / (n + self.codebook_size * self.epsilon) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.codebook.copy_(embed_normalized)

        # STE
        quantized = x + (quantized - x).detach()

        # Commitment loss (optional, can be much smaller)
        commitment_loss = F.mse_loss(x, quantized.detach())

        return quantized, commitment_loss, indices
```

#### B2. Residual EMA-VQ
```python
class ResidualEMAVQ(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.quantizers = nn.ModuleList([
            EMAVectorQuantizer(config) for _ in range(config.num_quantizers)
        ])
        self.num_active = config.num_quantizers

    def forward(self, x):
        residual = x
        quantized = torch.zeros_like(x)
        commitment_loss = 0.0
        codes = []

        for vq in self.quantizers[:self.num_active]:
            q, loss, indices = vq(residual)
            quantized = quantized + q
            residual = residual - q
            commitment_loss += loss
            codes.append(indices)

        return quantized, commitment_loss * 0.25, codes  # β=0.25
```

### 検証指標
- ✅ Perplexity Q0/Q1/Q2: ≥10
- ✅ Code usage: >10% (100+ codes active)
- ✅ RVQ Loss: 安定 (spike減少)
- ✅ Top1-Top2 distance margin: 縮小

### Config
- ファイル: `configs/phase_b_ema_vq.yaml`
- Base: Phase A結果
- 変更: EMA-VQ, β=0.25
- Steps: 10K

---

## Phase C: GAN導入（仕上げ）

### 目標
- モスキートーン除去
- 高域の質感改善
- 自然性向上

### 変更内容

#### C1. Multi-scale Discriminator (HiFi-GAN style)
```python
class MultiScaleDiscriminator(nn.Module):
    """HiFi-GAN style multi-scale discriminator."""
    def __init__(self):
        super().__init__()
        # 3 scales: original, /2, /4
        self.discriminators = nn.ModuleList([
            DiscriminatorP(),  # Period discriminators
            DiscriminatorS(),  # Scale discriminators
        ])

    def forward(self, x):
        ...
```

#### C2. Feature Matching Loss
```python
def feature_matching_loss(fmap_real, fmap_fake):
    loss = 0
    for dr, dg in zip(fmap_real, fmap_fake):
        for rl, gl in zip(dr, dg):
            loss += F.l1_loss(rl.detach(), gl)
    return loss
```

#### C3. Loss統合 with GAN warmup
```python
# Warmup schedule: 0-2K steps
adv_warmup = min(1.0, self.step / 2000.0)

total = (
    content_ce * 1.0
    + stft * 1.0
    + l1 * 10.0
    + rvq
    + rms * 0.1
    + multiband_rms * 0.05
    + adversarial * 1.0 * adv_warmup  # Warmup
    + feature_matching * 2.0 * adv_warmup
)
```

### 検証指標
- ✅ モスキートーン: 主観評価 (MOS)
- ✅ High-freq energy: 正常化
- ✅ ASR CER/WER: 元音声比で悪化なし
- ✅ STFT/L1: 維持または改善

### Config
- ファイル: `configs/phase_c_gan.yaml`
- Base: Phase B結果
- 追加: Multi-scale Discriminator
- Steps: 20K

---

## 各フェーズの期待結果

| Phase | Perplexity Q0 | RMS (vs target) | out_proj norm | モスキートーン |
|-------|---------------|-----------------|---------------|--------------|
| Phase 1 | 5-6 | ±2dB | 0.083 | 強い |
| Phase 2A | 8-10 → 6-8 | **-20dB (崩壊)** | **0.022 (崩壊)** | 強い |
| **Phase A** | **8-10** | **±0.5dB** | **0.08 (維持)** | 強い |
| **Phase B** | **12-15** | ±0.5dB | 0.08 | 中程度 |
| **Phase C** | 12-15 | ±0.5dB | 0.08 | **弱い** |

---

## 実行順序

1. **Phase A実装 + 5K training** (1-2日)
   - out_proj崩壊を確認
   - RMS監督の効果確認

2. **Phase A検証** (数時間)
   - 振幅が正常か
   - out_projノルム維持か
   - Perplexity Phase 1並みか

3. **Phase B実装 + 10K training** (2-3日)
   - EMA-VQ効果確認
   - Perplexity改善を確認

4. **Phase B検証**
   - Code usage >10%
   - モスキートーン軽減したか

5. **Phase C実装 + 20K training** (3-4日)
   - GAN導入
   - 最終品質評価

---

## KPI追跡 (TensorBoard)

### 各ステップで記録
- `train/out_proj_weight_norm`: out_proj weight のL2ノルム
- `train/out_proj_bias_norm`: out_proj bias のL2ノルム
- `train/rms_loss`: Frame RMS loss
- `train/multiband_rms_loss`: Multi-band RMS loss
- `train/audio_rms_mean`: 生成音声のRMS平均
- `train/audio_rms_std`: 生成音声のRMS標準偏差
- `train/high_freq_ratio`: >6kHz energy ratio
- `train/rvq_perplexity_q{i}`: 各quantizerのPerplexity
- `train/rvq_usage_q{i}`: Code usage ratio
- `train/rvq_top1_top2_margin`: Top1とTop2の距離差

### 判定基準
- ✅ **Phase A成功**: out_proj_weight_norm > 0.07, audio_rms ±1dB
- ✅ **Phase B成功**: perplexity_q0 > 10, usage_q0 > 0.10
- ✅ **Phase C成功**: 主観評価MOS > 3.5, モスキートーン軽減

---

## 参考文献

1. **SoundStream**: https://arxiv.org/abs/2107.03312
2. **VQ-VAE v2**: https://arxiv.org/abs/1906.00446
3. **HiFi-GAN**: https://arxiv.org/abs/2010.05646
4. **EnCodec**: https://arxiv.org/abs/2210.13438

---

## Notes

- **Cosine VQは完全に破棄**（Phase 2Aの失敗を踏まえ）
- **Progressive RVQは継続**（効果は確認済み）
- **各フェーズは2K刻みでAB比較**（早期検出）
- **必ずPhase A→B→Cの順に実施**（依存関係あり）
