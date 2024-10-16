import os
import torch
import numpy as np
import MeCab
from LSTM import Seq2Seq, Encoder, Decoder, DotProductAttention
from cbow import read_vocab, read_sentences
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from train import TranslationDataset, collate_fn
from torch.utils.data import DataLoader, Dataset
import nltk
import torch.nn as nn
import torch.optim as optim

# 设置随机种子以保证结果可重复
SEED = 2024
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 读取词汇表
eng_word2id, eng_id2word, eng_size = read_vocab("eng_vocab.txt")
jpn_word2id, jpn_id2word, jpn_size = read_vocab("jpn_vocab.txt")

class TranslationDataset(Dataset):
    def __init__(self, file_path, jpn_word2id, eng_word2id):
        self.data = []
        m = MeCab.Tagger("-Owakati")
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                src, trg = line.strip().split("\t")
                src = m.parse(src).strip().split()
                trg = nltk.word_tokenize(trg)
               
                src = ["<sos>"] + src + ["<eos>"]
                trg = ["<sos>"] + trg + ["<eos>"]
                self.data.append((src, trg))
        self.jpn_word2id = jpn_word2id
        self.eng_word2id = eng_word2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, trg = self.data[idx]
        src_indices = [self.jpn_word2id.get(word, self.jpn_word2id["<unk>"]) for word in src]
        trg_indices = [self.eng_word2id.get(word, self.eng_word2id["<unk>"]) for word in trg]
        return torch.tensor(src_indices, dtype=torch.long), torch.tensor(trg_indices, dtype=torch.long)

def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_lens = [len(src) for src in src_batch]
    trg_lens = [len(trg) for trg in trg_batch]
    
    src_padded = nn.utils.rnn.pad_sequence(src_batch, padding_value=jpn_word2id["<pad>"], batch_first=False)
    trg_padded = nn.utils.rnn.pad_sequence(trg_batch, padding_value=eng_word2id["<pad>"], batch_first=False)
    
    return src_padded, trg_padded

def translate(sentence):
    # 处理输入句子
    m = MeCab.Tagger("-Owakati")
    sentence = m.parse(sentence).strip().split()
    sentence = ["<sos>"] + sentence + ["<eos>"]
    # print(sentence)
    src_indices = [jpn_word2id.get(word, jpn_word2id["<unk>"]) for word in sentence]
    src_tensor = torch.tensor(src_indices, dtype=torch.long).unsqueeze(1).to(device)
    # print("src_tensor1", src_tensor)
    # print(src_tensor.shape)
    
    # 初始化模型
    encoder = Encoder(jpn_size, 256, 1024, 1).to(device)
    attention = DotProductAttention().to(device)
    decoder = Decoder(eng_size, 256, 1024, attention, 1).to(device)
    model = Seq2Seq(encoder, decoder, device).to(device)
    model.eval()
    model.load_state_dict(torch.load("seq2seq.pth", map_location=device, weights_only=True))
    with torch.no_grad():
        outputs = model(src_tensor, None, 0)  # [trg_len, batch_size, output_dim]
        # print(outputs)
        # print([eng_id2word[idx.item()] for idx in outputs])
        pass
    return [eng_id2word[idx.item()] for idx in outputs]

def calculate_bleu_score(references, candidates):
    '''
    计算BLEU分数
    reference: 参考翻译 二维列表
    candidate: 候选翻译 二维列表
    '''
    total_score = 0
    for ref, cand in zip(references, candidates):
        # print([ref])
        # print(cand)
        smooth = SmoothingFunction()
        score = sentence_bleu([ref], cand, smoothing_function=smooth.method1)
        total_score += score
    return total_score / len(references)

def calculate_perplexity(model, dataloader):
    criterion = nn.CrossEntropyLoss(ignore_index=eng_word2id["<pad>"])
    total_loss = 0
    total_words = 0
    for src, trg in dataloader:
        src = src.to(device)
        trg = trg.to(device)
        output = model(src, trg, 0)  # [trg_len, batch_size, output_dim]
        output_dim = output.shape[2]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        total_loss += loss.item()
        total_words += trg.shape[0]
    return np.exp(total_loss / total_words)

def calculate_bleu(model, dataloader):
    references = []
    candidates = []
    for src, trg in dataloader:
        src = src.to(device)
        trg = trg.to(device)
        for i in range(trg.shape[1]):
            reference = [eng_id2word[idx.item()] for idx in trg[:, i]]
            # 去除<sos>, <eos>, <pad>
            reference = [word for word in reference if word not in ["<sos>", "<eos>", "<pad>"]]
            references.append(reference)
        output = model(src, trg, 0)
        output = output.argmax(dim=2)
        for i in range(output.shape[1]):
            candidate = [eng_id2word[idx.item()] for idx in output[:, i]]
            # 去除<sos>, <eos>, <pad>
            candidate = [word for word in candidate if word not in ["<sos>", "<eos>", "<pad>"]]
            candidates.append(candidate)
    return calculate_bleu_score(references, candidates)

if __name__ == "__main__":
    # jpn_sentences, eng_sentences = read_sentences("test.txt")
    test_dataset = TranslationDataset("test.txt", jpn_word2id, eng_word2id)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
    references = []
    candidates = []
    encoder = Encoder(jpn_size, 256, 1024, 1).to(device)
    attention = DotProductAttention().to(device)
    decoder = Decoder(eng_size, 256, 1024, attention, 1).to(device)
    model = Seq2Seq(encoder, decoder, device).to(device)
    model.eval()
    model.load_state_dict(torch.load("seq2seq.pth", map_location=device, weights_only=True))
    print("Perplexity on test dataset: ", calculate_perplexity(model, test_dataloader))
    print("BLEU score on test dataset: ", calculate_bleu(model, test_dataloader))

    print(translate("私の名前は愛です。"))
    print(translate("昨日はお肉を食べません。"))
    print(translate("いただきますよう。"))
    print(translate("秋は好きです。"))
    print(translate("おはようございます。"))