# LSTMを使用した機械翻訳モデル
## 説明

LSTMを使用した機械翻訳モデル

## 要件

- Python 3.7.3
- PyTorch 1.6.0

## 使用方法

- 学習
```
$ train.py \
    --src_vocab_path [source vocabulary] \
    --tgt_vocab_path [target vocabulary] \
    --src_train_path [source train data] \
    --tgt_train_path [target train data] \
    --src_valid_path [source validation data] \
    --tgt_valid_path [target validation data] \
    --sentence_num 20000 \
    --max_length 50 \
    --batch_size 50 \
    --epoch_size 20 \
    --hidden_size 256 \
    --learning_rate 0.01 \
    --name [model name] \
    --seed 42 \
```

- 評価
```
$ eval.py \
    [model name]/model_state.pt \
    --src_eval_path [source evaluation data] \
    --batch_size 50 \
    --name [output file name]
```
