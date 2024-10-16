import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np
import MeCab
from LSTM import Encoder, Decoder, Seq2Seq, DotProductAttention
from build_vocab import eng_tokenize
import nltk

# 设置随机种子以保证结果可重复
SEED = 2024
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 读取数据集
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

# 读取词汇表
def read_vocab(vocab_file):
    with open(vocab_file, "r", encoding="utf-8") as f:
        vocab = f.readlines()
    word2id = {word.strip(): idx for idx, word in enumerate(vocab)}
    return word2id

# collate_fn 用于动态填充批次中的序列
def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_lens = [len(src) for src in src_batch]
    trg_lens = [len(trg) for trg in trg_batch]
    
    src_padded = nn.utils.rnn.pad_sequence(src_batch, padding_value=jpn_word2id["<pad>"], batch_first=False)
    trg_padded = nn.utils.rnn.pad_sequence(trg_batch, padding_value=eng_word2id["<pad>"], batch_first=False)
    
    return src_padded, trg_padded

# 读取数据
train_file = 'train.txt'
val_file = 'val.txt'
jpn_vocab_file = 'jpn_vocab.txt'
eng_vocab_file = 'eng_vocab.txt'

# 读取词汇表
jpn_word2id = read_vocab(jpn_vocab_file)
eng_word2id = read_vocab(eng_vocab_file)
# 创建数据集
train_dataset = TranslationDataset(train_file, jpn_word2id, eng_word2id)
val_dataset = TranslationDataset(val_file, jpn_word2id, eng_word2id)

BATCH_SIZE = 128
train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_iterator = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# 模型参数
INPUT_DIM = len(jpn_word2id)
OUTPUT_DIM = len(eng_word2id)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 1024
N_LAYERS = 1

# 读取预训练embedding模型参数
jpn_embedding = torch.load("cbow_jpn.pth", weights_only=True)['embedding.weight']
eng_embedding = torch.load("cbow_eng.pth", weights_only=True)['embedding.weight']

# 初始化模型
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, jpn_embedding)
attn = DotProductAttention()
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, attn, N_LAYERS, eng_embedding)
model = Seq2Seq(enc, dec, device).to(device)

# 初始化优化器和损失函数
optimizer = optim.Adam(model.parameters())
TRG_PAD_IDX = eng_word2id["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

# 训练函数
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for i, (src, trg) in enumerate(iterator):
        src = src.to(device)
        trg = trg.to(device)
        
        optimizer.zero_grad()
        
        output = model(src, trg, 0.4)
        
        # trg shape: [trg_len, batch_size]
        # output shape: [trg_len, batch_size, output_dim]
        output_dim = output.shape[-1]
        
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

# 验证函数
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src = src.to(device)
            trg = trg.to(device)
            
            output = model(src, trg, 0)  # 关掉teacher forcing
            
            # trg shape: [trg_len, batch_size]
            # output shape: [trg_len, batch_size, output_dim]
            output_dim = output.shape[-1]
            
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            
            loss = criterion(output, trg)
            epoch_loss += loss.item()
            
            # 计算准确率
            predictions = output.argmax(1)
            total_correct += (predictions == trg).sum().item()
            total_count += trg.size(0)
    
    accuracy = total_correct / total_count
    return epoch_loss / len(iterator), accuracy

# 开始训练
N_EPOCHS = 10
CLIP = 1
PATIENCE = 30
best_valid_loss = float('inf')
patience_counter = 0

print("Start training...")

for epoch in range(N_EPOCHS):
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss, valid_accuracy = evaluate(model, val_iterator, criterion)
    
    print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Val Loss: {valid_loss:.3f} | Val Accuracy: {valid_accuracy:.2f}')
    
    # Early stopping
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping...")
            break

# 保存模型
torch.save(model.state_dict(), 'seq2seq.pth')
print("Model saved!")