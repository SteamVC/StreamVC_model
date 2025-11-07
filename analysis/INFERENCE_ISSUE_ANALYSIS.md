# Phase 1-EMA-GAN 推論品質問題の分析

## 問題の概要

Step 4000のチェックポイントで推論を実行したところ、生成音声がノイズのみで、全く音声として認識できない状態。

## 音声統計の比較

### SOURCE (変換元音声)
```
RMS: 0.074685
Peak: 0.479126
Mean: -0.000005
Std: 0.074685
Clipped: 0 (0.00%)
Samples: 117,121

Amplitude distribution:
  [0.0, 0.1):  84.9%
  [0.1, 0.3):  14.8%
  [0.3, 0.5):   0.4%
  [0.5, 0.7):   0.0%
  [0.7, 0.9):   0.0%
  [0.9, 1.0):   0.0%
```
→ **正常な音声**: RMS 0.075、ピーク 0.48、ほとんどが低振幅域に分布

### TARGET (ターゲット参照音声)
```
RMS: 0.029738
Peak: 0.253021
Mean: -0.000085
Std: 0.029738
Clipped: 0 (0.00%)
Samples: 23,762

Amplitude distribution:
  [0.0, 0.1):  98.2%
  [0.1, 0.3):   1.8%
  [0.3, 0.5):   0.0%
  [0.5, 0.7):   0.0%
  [0.7, 0.9):   0.0%
  [0.9, 1.0):   0.0%
```
→ **正常な音声**: RMS 0.030、ピーク 0.25、非常に低振幅域に集中

### GENERATED (生成音声 - Step 4000)
```
RMS: 0.318035
Peak: 1.000000
Mean: -0.239720
Std: 0.208998
Clipped: 1 (0.01%)
Samples: 15,720

Amplitude distribution:
  [0.0, 0.1):   7.1%
  [0.1, 0.3):  27.0%
  [0.3, 0.5):  65.1%  ← 異常に高い振幅域に集中
  [0.5, 0.7):   0.9%
  [0.9, 1.0):   0.0%
```
→ **異常な音声**:
- RMS 0.318 (SOURCE比4.3倍、TARGET比10.7倍)
- ピーク 1.0 (完全にクリッピング)
- Mean -0.24 (本来0付近であるべき)
- 65.1%が[0.3, 0.5)域に集中 (正常音声は0.4%以下)

## 学習時との比較

### Training時の統計 (Step 4000)
```
Audio RMS: 0.032
Pre-RVQ Std: 1.000  # 正規化により1.0に固定
RVQ Loss: 0.010
```

### Inference時の統計
```
Audio RMS: 0.318  (学習時の10倍！)
RVQ Loss: 0.065
```

## 異常な挙動の分析

### 1. スケールの爆発
- 学習時: Audio RMS 0.032
- 推論時: Audio RMS 0.318 (10倍)
- **原因候補**: RVQの正規化とスケール復元の不一致

### 2. DCオフセットの発生
- Mean: -0.240 (本来0付近)
- **原因候補**: Batch Normalization的な統計の不一致

### 3. 振幅分布の異常
- 正常音声: 85%以上が[0.0, 0.1)域
- 生成音声: 65%が[0.3, 0.5)域
- **症状**: ホワイトノイズに近い分布

## 推論スクリプトの検証

### `/Users/akatuki/streamVC/scripts/infer.py`の処理フロー

```python
# 1. チェックポイント読み込み
pipeline.load_state_dict(checkpoint["model"])
pipeline.eval()

# 2. 音声前処理
waveform = torch.FloatTensor(waveform).unsqueeze(0)  # (1, T)

# 3. 話者埋め込みエンコード
pipeline.encode_speaker(target_ref_audio)

# 4. ピッチ統計リセット
pipeline.reset_pitch_stats()

# 5. 推論実行
with torch.no_grad():
    outputs = pipeline(source_audio, mode="infer")

# 6. 音声保存
generated = outputs["audio"].cpu().squeeze(0).numpy()
sf.write(output_path, generated, sample_rate)
```

**確認ポイント**:
- ✅ `pipeline.eval()` でBatchNormやDropoutは無効化
- ✅ `mode="infer"` でピッチ抽出器も推論モード
- ✅ `with torch.no_grad()` で勾配計算は無効化
- ❓ RVQの正規化とスケール復元の挙動

## RVQ処理フローの検証

### Decoder内のRVQ前処理 (`decoder.py:122-137`)
```python
# Pre-RVQ processing: 1x1 conv + manual normalization
x_for_rvq = self.pre_rvq_conv(x)  # (B, C, T)

# Manual normalization: mean=0, std=1
x_for_rvq = x_for_rvq - x_for_rvq.mean(dim=-1, keepdim=True)
x_for_rvq = x_for_rvq / (x_for_rvq.std(dim=-1, keepdim=True) + 1e-5)

# Transpose for RVQ: (B, C, T) -> (B, T, C)
x_for_rvq = x_for_rvq.transpose(1, 2)

quantized, rvq_loss, codes = self.rvq(x_for_rvq)
audio = self.out_proj(quantized.transpose(1, 2)).squeeze(1)
```

### RVQ内の処理 (`rvq.py:67-127`)
```python
# L2 distance VQ in normalized space
residual_flat = residual.reshape(-1, residual.shape[-1])  # (B*T, C)
distances = residual_sq + codebook_sq.t() - 2 * dot_product
indices = torch.argmin(distances, dim=-1)
embeds = F.embedding(indices, codebook)

# Phase A: STE in same coordinate space (normalized)
embeds_st = residual + (embeds - residual).detach()
# Per-quantizer affine transform
embeds_scaled = embeds_st * self.post_scale[q_idx] + self.post_bias[q_idx]
quantized_sum = quantized_sum + embeds_scaled

# Final 1x1 conv for scale adjustment and channel mixing
quantized_final = self.final_conv(quantized_sum.transpose(1, 2)).transpose(1, 2)
```

## 疑わしい箇所

### 1. Batch統計の不一致 (最有力候補)
**問題**: 学習時と推論時でバッチサイズが異なる
- 学習時: `batch_size=8` → 統計が8サンプルで計算
- 推論時: `batch_size=1` → 統計が1サンプルで計算

**影響を受ける箇所**:
```python
# decoder.py:126-127
x_for_rvq = x_for_rvq - x_for_rvq.mean(dim=-1, keepdim=True)
x_for_rvq = x_for_rvq / (x_for_rvq.std(dim=-1, keepdim=True) + 1e-5)
```
- この正規化は**channel方向(dim=-1)**で行われている
- バッチサイズが1の場合、統計の不安定性が増す可能性

### 2. RVQ Codebookのスケール不整合
**問題**: EMAで更新されたCodebookが正規化空間で学習されている
- Codebookは正規化された空間(mean=0, std=1)で学習
- 推論時の入力が異なる統計を持つ場合、量子化誤差が増大

### 3. `post_scale`と`post_bias`の学習不足
**問題**: Per-quantizer affine transformが学習不十分
```python
# rvq.py:108
embeds_scaled = embeds_st * self.post_scale[q_idx] + self.post_bias[q_idx]
```
- これらのパラメータが学習時のバッチ統計に過度に依存している可能性

### 4. `final_conv`の初期化問題
**問題**: Identity初期化が保持されていない可能性
```python
# rvq.py:44-47
self.final_conv = nn.Conv1d(config.dims, config.dims, kernel_size=1)
nn.init.eye_(self.final_conv.weight.squeeze())
nn.init.zeros_(self.final_conv.bias)
```
- 学習により大きく変化した場合、スケールが崩れる

## 次の調査ステップ

### 優先度A: バッチサイズ依存性の検証
1. **学習時の統計をロギング**:
   - `x_for_rvq`の正規化前後の統計
   - `quantized`のチャネルごとの統計
   - `audio`出力のRMS

2. **推論時の統計をロギング**:
   - 同じ箇所の統計を出力
   - 学習時との差分を確認

3. **Batch Normalization的な解決策**:
   - Running mean/stdを保存して推論時に使用
   - または正規化を無効化してスケール学習に任せる

### 優先度B: RVQパラメータの確認
1. **Codebookの統計**:
   - 各quantizerのcodebookのmean/std
   - EMA更新による変化

2. **`post_scale`/`post_bias`の値**:
   - 学習により大きく変化しているか
   - デフォルト(1.0/0.0)からどの程度離れているか

3. **`final_conv`のweight**:
   - Identity初期化から大きく変化しているか

### 優先度C: Pipeline全体の中間出力確認
1. **Content Encoder出力**: 正常か
2. **Pitch Extractor出力**: 正常か
3. **Speaker Encoder出力**: 正常か
4. **Decoder (pre-RVQ)出力**: 正常か
5. **RVQ出力**: ここで異常が発生している可能性

## 仮説

**最有力仮説**: バッチサイズ1での正規化統計の不安定性
- 学習時: `batch_size=8`, サンプル長1.28秒 (20,480サンプル) → 正規化が安定
- 推論時: `batch_size=1`, サンプル長可変 → 正規化が不安定
- RVQのCodebookは学習時の統計分布に特化してしまっている

**解決策候補**:
1. **Running statistics導入**: BatchNorm的にRunning mean/stdを保存
2. **Layer Normalization化**: チャネル方向ではなく時間方向で正規化
3. **正規化の削除**: スケール学習をRVQ内のaffine transformに完全に任せる
4. **Inference時の統計補正**: 学習時の平均的な統計を使用
