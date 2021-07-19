import argparse
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import models
import dataset
from tqdm import tqdm
from train import load_sentences, convert_sent_to_word, convert_word_to_idx


def test(BOS, EOS, encoder, decoder, eval_loader, dictionary, max_len, file_name, device):

    encoder.eval()
    decoder.eval()
    pbar = tqdm(eval_loader, ascii=True)
    sentences = ""
        
    for batch in pbar:
        
        # source データ
        source = batch.to(device)

        # hidden, cell の初期化
        hidden = torch.zeros(source.size(0), encoder.hidden_size, device=device)
        cell   = torch.zeros(source.size(0), encoder.hidden_size, device=device)
        
        
        # ----- encoder へ入力 -----
        
        # 転置 (batch_size * words_num) --> (words_num * batch_size)
        source_t = torch.t(source)
        
        # source_words は入力単語を表す tensor (batch_size)
        for source_words in source_t:
            hidden, cell = encoder(source_words, hidden, cell)

            
        # ----- decoder へ入力 -----
        
        # 最初の入力は [BOS]
        input_words = torch.tensor([BOS] * source.size(0), device=device)

        # 最大文長を max_len + 50 として出力
        batch_words = [[] for _ in range(source.size(0))]
        states = [True for _ in range(source.size(0))]
        for _ in range(max_len + 50):
            output, hidden, cell = decoder(input_words, hidden, cell)
                
            # output (batch_size * dict_size)
            predicted_words = torch.argmax(output, dim=1)
            
            input_words = predicted_words
            
            # batch_words に単語を格納
            for i in range(source.size(0)):
                token_idx = predicted_words[i].item()
                if states[i] == True:
                    if token_idx == EOS:
                        states[i] = False
                    else:
                        batch_words[i].append(dictionary[token_idx])

        # 文章を生成
        for words in batch_words:
            sentences += " ".join(words) + "\n"
                
        pbar.set_description("[sentence generation]")

    # 文章をファイルに出力
    with open(file_name, mode="w") as output_f:
        output_f.write(sentences)

    print("Fin.")



def main():

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--src_eval_path", type=str, default="../corpus/ASPEC-JE/corpus.tok/test.en")
    
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("model_name", type=str)
    parser.add_argument("--name", type=str, default="output")
    
    args = parser.parse_args()
    
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    
    model_dir = os.path.dirname(args.model_name)
    with open("{}/config.json".format(model_dir)) as f:
        config = json.load(f)
    config = argparse.Namespace(**config)
    
    # データのロード
    src_dict_data = torch.load(config.src_dict_path)
    tgt_dict_data = torch.load(config.tgt_dict_path)
    src2idx = src_dict_data["dict"]["word2index"]
    idx2tgt = tgt_dict_data["dict"]["index2word"]
    PAD = src2idx["[PAD]"]
    BOS = src2idx["[BOS]"]
    EOS = src2idx["[EOS]"]
    src_dict_size = len(src2idx)
    tgt_dict_size = len(idx2tgt)
    
    # load eval data
    src_eval_sent_list = load_sentences(args.src_eval_path)
    
    # convert sent to word
    src_eval_word_list = convert_sent_to_word(src_eval_sent_list)
    
    # convert word to idx
    src_eval_idx_list = convert_word_to_idx(word_list=src_eval_word_list, word2index=src2idx)
    
    eval_data = dataset.SingleDataset(src_data=src_eval_idx_list)
    eval_loader = DataLoader(eval_data, batch_size=args.batch_size, collate_fn=dataset.collate_fn, shuffle=False)
    
    # モデルのロード
    print("model:", args.model_name)
    
    # モデルの読込
    model_states = torch.load(args.model_name)
    encoder = models.EncoderLSTM(PAD, config.hidden_size, src_dict_size).to(device)
    decoder = models.DecoderLSTM(PAD, config.hidden_size, tgt_dict_size).to(device)
    encoder.load_state_dict(model_states["encoder_state"])
    decoder.load_state_dict(model_states["decoder_state"])
    
    # 文生成
    test(BOS, EOS, encoder, decoder, eval_loader, idx2tgt, config.max_length,  "./output.tok", device)



if __name__ == "__main__":
    main()
