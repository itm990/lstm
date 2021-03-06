# LSTM

## 説明

LSTMを使用した機械翻訳の実装です．
原言語，目的言語の語彙データ，トーカナイズ済み対訳文データを使用して，原言語文から目的言語文への翻訳の学習，推論を行います．

## 要件

- Python 3.7.3
- PyTorch 1.6.0
- tqdm 4.56.0
- nltk 3.4.3

## 使用方法

- 学習

```
$ train.py \
    --src_vocab_path [source vocabulary data] \
    --tgt_vocab_path [target vocabulary data] \
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
    --seed 42
```

- 推論

```
$ test.py \
    [model name]/model_state.pt \
    --src_eval_path [source evaluation data] \
    --batch_size 50 \
    --name [output file name]
```
