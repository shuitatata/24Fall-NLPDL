# 使用cbow模型 训练word2vec
import os
import numpy as np
import torch
import torch.nn as nn
import MeCab

def read_vocab(vocab_file):
    with open(vocab_file, "r", encoding="utf-8") as f:
        vocab = f.readlines()
    
    word2id = {word.strip(): idx for idx, word in enumerate(vocab)}
    id2word = {idx: word.strip() for idx, word in enumerate(vocab)}
    size = len(vocab)
    return word2id, id2word, size

def read_sentences(file):
    with open(file, "r", encoding="utf-8") as f:
        data = f.readlines()
        jpn_sentences = [line.split("\t")[0] for line in data]
        eng_sentences = [line.split("\t")[1].split() for line in data]
    return jpn_sentences, eng_sentences

def build_jpn_data(sentences, word2id, window_size=2):
    data = []
    m = MeCab.Tagger("-Owakati")
    for sentence in sentences:
        # print(sentence)
        sentence = m.parse(sentence).split()
        for i, word in enumerate(sentence):
            context_words = sentence[max(i-window_size, 0):i] + sentence[i+1:min(i+window_size+1, len(sentence))]
            data.append((word, context_words))
    return data

def build_eng_data(sentences, word2id, window_size=2):
    data = []
    for sentence in sentences:
        for i, word in enumerate(sentence):
            context_words = sentence[max(i-window_size, 0):i] + sentence[i+1:min(i+window_size+1, len(sentence))]
            data.append((word, context_words))
    return data

def one_hot_encoding(word, word2id):
    vec = torch.zeros(len(word2id))
    vec[word2id[word]] = 1
    return vec

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256):
        super(CBOW, self).__init__()
        self.linear_input = nn.Linear(vocab_size, embedding_dim, bias=False)
        self.linear_output = nn.Linear(embedding_dim, vocab_size, bias=False)
    
    def forward(self, x):
        x = self.linear_input(x)
        x = torch.mean(x, dim=0)
        x = self.linear_output(x)
        return torch.log_softmax(x, dim=0)

def train(model, data, word2id, id2word, language, epochs=10, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    for epoch in range(epochs):
        total_loss = 0
        for word, context_words in data:
            # print([one_hot_encoding(word, word2id) for word in context_words])
            X = torch.stack([one_hot_encoding(word, word2id) for word in context_words])
            y = torch.tensor(word2id[word])
            optimizer.zero_grad()
            output = model(X)
            loss = loss_fn(output, y)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print("Epoch: {}, Loss: {}".format(epoch, total_loss/len(data)))
    # 保存模型
    torch.save(model.state_dict(), f"cbow_{language}.pth")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 读取词汇表
    jpn_word2id, jpn_id2word, jpn_size = read_vocab("jpn_vocab.txt")
    eng_word2id, eng_id2word, eng_size = read_vocab("eng_vocab.txt")
    # 读取数据
    jpn_sentences, eng_sentences = read_sentences("train.txt")

    # print(jpn_sentences[:10])

    # 构建数据
    jpn_data = build_jpn_data(jpn_sentences, jpn_word2id)
    eng_data = build_eng_data(eng_sentences, eng_word2id)

    jp_model = CBOW(jpn_size).to(device)
    en_model = CBOW(eng_size).to(device)

    jpn_data = jpn_data.to(device)
    eng_data = eng_data.to(device)

    train(jp_model, jpn_data, jpn_word2id, jpn_id2word, "jpn")
    train(en_model, eng_data, eng_word2id, eng_id2word, "eng")
    print("Done")


    
    
    