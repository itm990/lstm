import argparse
import datetime
import json
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
import dataset
import models
import validate
from preprocess import make_dict, load_sentences, convert_sent_to_word, trim_list, convert_word_to_idx


def train(BOS, EOS, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, epoch_size,
          train_loader, valid_loader, valid_word_data, dictionary, max_len, device):

    print("start training.")
    
    for epoch in range(epoch_size):

        encoder.train()
        decoder.train()
        pbar = tqdm(train_loader, ascii=True)
        total_loss = 0

        for i, batch in enumerate(pbar):

            # 初期化
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss = 0

            # データの分割
            enc_in, dec_in, dec_out = map(lambda x: x.to(device), batch)
            
            # hidden, cell の初期化
            hidden = torch.zeros(enc_in.size(0), encoder.hidden_size, device=device)
            cell   = torch.zeros(enc_in.size(0), encoder.hidden_size, device=device)


            # ----- encoder へ入力 -----
            
            # 転置 (batch_size * words_num) --> (words_num * batch_size)
            enc_in_t = torch.t(enc_in)

            # source_words は長さ batch_size の 1 次元 tensor
            for source_words in enc_in_t:
                hidden, cell = encoder(source_words, hidden, cell)

                
            # ----- decoder へ入力 -----
            
            # 転置 (batch_size * words_num) --> (words_num * batch_size)
            dec_in_t = torch.t(dec_in)
            dec_out_t = torch.t(dec_out)

            # target_words は長さ batch_size の 1 次元 tensor
            for in_words, out_words in zip(dec_in_t, dec_out_t):
                output, hidden, cell = decoder(in_words, hidden, cell)
                
                # 損失の計算
                loss += criterion(output, out_words)

            total_loss += loss
            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()
            
            pbar.set_description("[epoch:%d] loss:%f" % (epoch+1, total_loss/(i+1)))

        bleu = validate.validate(BOS, EOS, encoder, decoder, valid_loader, valid_word_data, dictionary, max_len, device)
        print("BLEU:", bleu)

    print("Fin.")



def main():
    
    datetime_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # パラメータの設定
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--src_vocab_path", type=str, default=None)
    parser.add_argument("--tgt_vocab_path", type=str, default=None)
    parser.add_argument("--src_train_path", type=str, default=None)
    parser.add_argument("--tgt_train_path", type=str, default=None)
    parser.add_argument("--src_valid_path", type=str, default=None)
    parser.add_argument("--tgt_valid_path", type=str, default=None)
    parser.add_argument("--sentence_num", type=int, default=20000)
    parser.add_argument("--max_length", type=int, default=50)
    
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--epoch_size", type=int, default=20)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--name", type=str, default="no_name")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    
    # save config file
    save_dir = "./model/{}_{}".format(args.name, datetime_str) if args.name != "no_name" else "./model/no_name"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open("{}/config.json".format(save_dir), mode="w") as f:
        json.dump(vars(args), f, separators=(",", ":"), indent=4)
    
    # データのロード
    src2idx, idx2src = make_dict(args.src_vocab_path)
    tgt2idx, idx2tgt = make_dict(args.tgt_vocab_path)
    PAD = src2idx["[PAD]"]
    BOS = src2idx["[BOS]"]
    EOS = src2idx["[EOS]"]
    src_dict_size = len(src2idx)
    tgt_dict_size = len(tgt2idx)
    
    # load train data
    src_train_sent_list = load_sentences(args.src_train_path)
    tgt_train_sent_list = load_sentences(args.tgt_train_path)
    src_valid_sent_list = load_sentences(args.src_valid_path)
    tgt_valid_sent_list = load_sentences(args.tgt_valid_path)
    
    # convert sent to word
    src_train_word_list = convert_sent_to_word(src_train_sent_list)
    tgt_train_word_list = convert_sent_to_word(tgt_train_sent_list)
    src_valid_word_list = convert_sent_to_word(src_valid_sent_list)
    tgt_valid_word_list = convert_sent_to_word(tgt_valid_sent_list)
    
    # trim word list
    src_train_word_list, tgt_train_word_list = trim_list(
        src_train_word_list, tgt_train_word_list, sent_num=args.sentence_num, max_len=args.max_length
    )
    
    # convert word to idx
    src_train_idx_list = convert_word_to_idx(word_list=src_train_word_list, word2index=src2idx)
    tgt_train_idx_list = convert_word_to_idx(word_list=tgt_train_word_list, word2index=tgt2idx)
    src_valid_idx_list = convert_word_to_idx(word_list=src_valid_word_list, word2index=src2idx)
    
    train_data = dataset.PairedDataset(bos_idx=BOS, eos_idx=EOS, src_data=src_train_idx_list, tgt_data=tgt_train_idx_list)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, collate_fn=dataset.paired_collate_fn, shuffle=True)
    valid_data = dataset.SingleDataset(src_data=src_valid_idx_list)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, collate_fn=dataset.collate_fn, shuffle=False)
    valid_word_data = [ [words] for words in tgt_valid_word_list ]
    
    # 設定
    encoder = models.EncoderLSTM(PAD, args.hidden_size, src_dict_size).to(device)
    decoder = models.DecoderLSTM(PAD, args.hidden_size, tgt_dict_size).to(device)
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=args.learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=args.learning_rate)
    criterion = nn.NLLLoss(ignore_index=PAD, reduction="sum")
    
    # 学習
    train(BOS, EOS, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, args.epoch_size,
          train_loader, valid_loader, valid_word_data, idx2tgt, args.max_length, device)
    
    # モデル状態の保存
    model_states = {
        "encoder_state": encoder.state_dict(),
        "decoder_state": decoder.state_dict()
    }
    torch.save(model_states, "{}/model_state.pt".format(save_dir))
    print("model_name:", args.name)



if __name__ == "__main__":
    main()
