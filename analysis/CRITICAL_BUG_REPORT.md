# 致命的バグ: Progressive RVQの推論時不整合

## 問題の本質

**学習時**: Progressive RVQ により、Step 4000では3つのQuantizer (Q0, Q1, Q2)が有効
**推論時**: `num_active_quantizers`がチェックポイントに保存されず、デフォルト値8が使用される
**結果**: 未学習のCodebook (Q2-Q7)が推論時に使用され、ノイズが生成される

## 証拠

### 1. Codebook使用状況 (Step 4000)
```
Q0: 532/1024 codes used (Dead: 867)  ← 学習済み
Q1: 1024/1024 codes used (Dead: 593) ← 学習済み
Q2: 0/1024 codes used (Dead: 1024)   ← 未使用 (本来はstep 4000で有効化されるはず)
Q3-Q7: 0/1024 codes used             ← 未使用
```

### 2. Codebookの統計異常
```
Codebook 0: std=1.5078, min=-72.19, max=69.56  ← 異常に大きい範囲
Codebook 1: std=0.5968, min=-36.20, max=31.69  ← やや大きい
Codebook 2-7: std≈1.0, 範囲≈[-4, 4]            ← 初期化状態(ランダム)
```
→ **Codebook 2以降は初期化状態のまま**

### 3. 推論時の設定
```python
pipeline.decoder.rvq.num_active_quantizers = 8  ← デフォルト値
```
→ **未学習のCodebook 2-7も使用されてしまう**

### 4. Progressive RVQの計算式
```python
# trainer.py:168
num_active = min(self.step // self.config.rvq.progressive_steps + 1, self.config.num_quantizers)
# Step 4000: min(4000 // 2000 + 1, 8) = min(3, 8) = 3
```
→ **本来はQ0, Q1, Q2の3つが有効なはず**

## なぜQ2も未使用なのか？

**仮説1**: EMA更新のバグ
- cluster_sizeがゼロのまま → EMA更新が機能していない可能性
- `ema_update()`の実装に問題がある？

**仮説2**: Progressive RVQのタイミング問題
- Step 4000でちょうど3つ有効になったばかり
- EMA更新が遅れている
- または、forward時の`active_codebooks`の取得が間違っている

実際のコード確認:
```python
# rvq.py:75
active_codebooks = self.codebooks[:self.num_active_quantizers]
```
→ この行は正しい

**確定**: Trainerで`set_num_active_quantizers()`が呼ばれていない可能性

## Trainerの確認が必要

`trainer.py`で以下を確認:
1. `set_num_active_quantizers()`が毎ステップ呼ばれているか
2. チェックポイント保存時に`num_active_quantizers`を保存しているか
3. チェックポイント読み込み時に`num_active_quantizers`を復元しているか

## Codebook 0の異常なスケール

```
std=1.5078 (正常は1.0付近)
min=-72.19, max=69.56 (正常は±4付近)
```

**原因候補**:
1. **EMA初期化の問題**: 最初のcodebookだけEMA更新が異常に大きい変化を起こした
2. **Dead code resetの問題**: 867個のdead codesがリセットされる際に異常値が混入
3. **正規化の不整合**: 学習時の正規化前後でスケールが一致していない

**影響**:
- Codebook 0から取得される量子化ベクトルのスケールが異常に大きい
- `post_scale[0]=0.8901`で補正しようとしているが不十分
- 最終的な音声スケールの爆発につながる

## 修正方針

### 優先度S (即座に修正)

1. **チェックポイントに`num_active_quantizers`を保存**
```python
# trainer.py:save_checkpoint()
torch.save({
    'model': self.pipeline.state_dict(),
    'num_active_quantizers': self.pipeline.decoder.rvq.num_active_quantizers,  # 追加
    # ...
}, path)
```

2. **推論スクリプトでnum_active_quantizersを復元**
```python
# infer.py:load_checkpoint()
pipeline.load_state_dict(checkpoint["model"])
if "num_active_quantizers" in checkpoint:
    pipeline.decoder.rvq.set_num_active_quantizers(checkpoint["num_active_quantizers"])
```

3. **Trainerで毎ステップset_num_active_quantizersを呼ぶ**
```python
# trainer.py:train_step()
num_active = min(self.step // self.config.rvq.progressive_steps + 1, self.config.num_quantizers)
self.pipeline.decoder.rvq.set_num_active_quantizers(num_active)
```

### 優先度A (学習再開前に修正)

4. **Codebook 0のスケール異常の原因調査**
- EMA更新のロギング
- Dead code resetのロギング
- Codebookの統計変化を追跡

5. **EMA更新が正しく動作しているか検証**
- cluster_sizeが増加しているか
- embed_avgが更新されているか

### 優先度B (長期的な改善)

6. **Progressive RVQの検証テスト追加**
- 各ステップでの有効quantizer数を確認
- Codebook使用状況をロギング

7. **推論時の統計チェック**
- Codebookの範囲が異常な場合はwarning

## 現状のStep 4000チェックポイントの評価

**結論**: このチェックポイントは**推論に使用不可**

理由:
1. 実質2つのquantizerしか学習されていない(Q0, Q1のみ)
2. Q0のcodebookが異常なスケール(-72 ~ +69)
3. 推論時に未学習のQ2-Q7が使用されてしまう
4. 生成音声がノイズになるのは必然

**Step 4000での期待品質**:
- Progressive RVQが正しく動作していれば、3つのquantizer(Q0, Q1, Q2)で中程度の品質
- GANも4000ステップでは初期段階(warm-up完了、full powerで2000ステップのみ)
- まだ高品質な音声生成は期待できない段階

**次のアクション**:
1. 上記の修正を実装
2. 学習を再開 or 最初からやり直し
3. Step 8000-10000まで学習を進める(Q4-Q5まで有効化)
4. その時点で推論テストを再実行
