import torch
import torch.nn as nn
import torch.optim as optim
import configparser
from torch.utils.data import DataLoader
import dataset
from tqdm import tqdm
import models
import validate



def train(EOS, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, epoch_max,
          train_loader, valid_loader, valid_word_data, dictionary, max_len, device):

    print('start training.')
    
    for epoch in range(epoch_max):

        pbar = tqdm(train_loader, ascii=True)
        total_loss = 0

        for i, batch in enumerate(pbar):

            # 初期化
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss = 0

            # データの分割
            source, target = map(lambda x: x.to(device), batch)

            # hidden, cell の初期化
            hidden = torch.zeros(source.size(0), encoder.hidden_size, device=device)
            cell   = torch.zeros(source.size(0), encoder.hidden_size, device=device)


            # ----- encoder へ入力 -----
            
            # 転置 (batch_size * words_num) --> (words_num * batch_size)
            source_t = torch.t(source)

            # source_words は長さ batch_size の 1 次元 tensor
            for source_words in source_t:
                hidden, cell = encoder(source_words, hidden, cell)

                
            # ----- decoder へ入力 -----
            
            # 転置 (batch_size * words_num) --> (words_num * batch_size)
            target_t = torch.t(target)

            # 最初の入力は <EOS>
            input_words = torch.tensor([EOS] * source.size(0), device=device)

            # target_words は長さ batch_size の 1 次元 tensor
            for target_words in target_t:
                output, hidden, cell = decoder(input_words, hidden, cell)
                
                # 損失の計算
                loss += criterion(output, target_words)

                input_words = target_words
            
            total_loss += loss
            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()
            
            pbar.set_description('[epoch:%d] loss:%f' % (epoch+1, total_loss/(i+1)))

        bleu = validate.validate(EOS, encoder, decoder, valid_loader, valid_word_data, dictionary, max_len, device)
        print('BLEU:', bleu)

    print('Fin.')



def main():

    # パラメータの設定
    inifile = configparser.ConfigParser()
    inifile.read('./params_config.ini', 'UTF-8')
    batch_size = int(inifile.get('params', 'batch_size'))
    hidden_size = int(inifile.get('params', 'hidden_size'))
    learning_rate = float(inifile.get('params', 'learning_rate'))
    epoch_max = int(inifile.get('params', 'epoch_size'))

    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    
    # データのロード
    data_path = '/home/ikawa/tutorial/seq2seq/corpus/ASPEC-JE/corpus.tok/20000.dict'
    data = torch.load(data_path)
    PAD = data['word2index']['src']['<PAD>']
    EOS = data['word2index']['tgt']['<EOS>']
    max_len = data['settings'].max_word_seq_len
    idx2jpn = data['index2word']['tgt']
    src_dict_size = len(data['word2index']['src'])
    tgt_dict_size = len(data['word2index']['tgt'])
    train_data = dataset.PairedDataset(src_data=data['train']['src'], tgt_data=data['train']['tgt'])
    train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=dataset.paired_collate_fn, shuffle=True)
    valid_data = dataset.SingleDataset(src_data=data['valid']['src'])
    valid_loader = DataLoader(valid_data, batch_size=batch_size, collate_fn=dataset.collate_fn, shuffle=False)
    valid_tgt_word_data = data['valid']['tgt_word']
    valid_word_data = []
    for sentence in valid_tgt_word_data:
        valid_word_data.append([sentence[:-1]])

    # 設定
    encoder = models.EncoderLSTM(PAD, hidden_size, src_dict_size).to(device)
    decoder = models.DecoderLSTM(PAD, hidden_size, tgt_dict_size).to(device)
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss(ignore_index=PAD, reduction='sum')
    
    # 学習
    train(EOS, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, epoch_max,
          train_loader, valid_loader, valid_word_data, idx2jpn, max_len, device)
    
    # モデル状態の保存
    model_states = {
        'hidden_size'  : hidden_size,
        'src_dict_size': src_dict_size,
        'tgt_dict_size': tgt_dict_size,
        'encoder_state': encoder.state_dict(),
        'decoder_state': decoder.state_dict()
    }
    model_states_name = 'lstm_model'
    torch.save(model_states, 'model/{}.pt'.format(model_states_name))
    print('model_states_name:', model_states_name)



if __name__ == '__main__':
    main()
