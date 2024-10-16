import os
import numpy as np
import torch
import torch.nn as nn
import MeCab
from torch.utils.data import DataLoader, Dataset
from cbow import CBOW, CBOWDataset, read_vocab

if __name__ == "__main__":
    # 读取词汇表
    eng_word2id, eng_id2word, eng_size = read_vocab("eng_vocab.txt")

    # 初始化模型
    model = CBOW(eng_size)
    model.load_state_dict(torch.load("cbow_eng.pth",weights_only=True))
    model.eval()

    # 要测试的词汇
    word1 = "king"
    word2 = "apple"

    # 获取词向量
    word1_idx = torch.tensor(eng_word2id[word1], dtype=torch.long).unsqueeze(0)  # 加上 batch 维度
    word2_idx = torch.tensor(eng_word2id[word2], dtype=torch.long).unsqueeze(0)  # 加上 batch 维度

    word1_vec = model.embedding(word1_idx).squeeze(0).detach()  # 获取词向量并移除 batch 维度
    word2_vec = model.embedding(word2_idx).squeeze(0).detach()  # 获取词向量并移除 batch 维度

    # 输出词的索引
    print(eng_word2id[word1])
    print(eng_word2id[word2])

    # 计算余弦相似度
    cos_sim = torch.matmul(word1_vec, word2_vec)
    cos_sim = cos_sim / (torch.norm(word1_vec) * torch.norm(word2_vec))
    cos_sim = cos_sim.item()

    # 输出相似度
    print(f"Similarity between '{word1}' and '{word2}': {cos_sim:.4f}")
