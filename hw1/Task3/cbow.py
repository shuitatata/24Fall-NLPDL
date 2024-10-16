import os
import numpy as np
import torch
import torch.nn as nn
import MeCab
from torch.utils.data import DataLoader, Dataset
import nltk

class CBOWDataset(Dataset):
    def __init__(self, data, word2id):
        self.data = data
        self.word2id = word2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        word, context_words = self.data[idx]
        # 将context_words转换为词汇索引
        context = torch.tensor([self.word2id.get(cw, self.word2id["<unk>"]) for cw in context_words], dtype=torch.long)
        target = torch.tensor(self.word2id.get(word, self.word2id["<unk>"]), dtype=torch.long)
        return context, target

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
        eng_sentences = [line.split("\t")[1] for line in data]
        eng_sentences = [nltk.word_tokenize(sentence) for sentence in eng_sentences]
    return jpn_sentences, eng_sentences

def build_jpn_data(sentences, word2id, window_size=3):
    data = []
    m = MeCab.Tagger("-Owakati")
    for sentence in sentences:
        sentence = m.parse(sentence).split()
        # 加入<sos>和<eos>
        sentence = ["<sos>"] + sentence + ["<eos>"]
        for i, word in enumerate(sentence):
            context_words = sentence[max(i - window_size, 0):i] + sentence[i + 1:min(i + window_size + 1, len(sentence))]
            # 如果长度不够，用"<pad>"补齐
            while len(context_words) < 2 * window_size:
                context_words.append("<pad>")
            data.append((word, context_words))
    return data

def build_eng_data(sentences, word2id, window_size=2):
    data = []
    for sentence in sentences:
        # 加入<sos>和<eos>
        sentence = ["<sos>"] + sentence + ["<eos>"]
        for i, word in enumerate(sentence):
            context_words = sentence[max(i - window_size, 0):i] + sentence[i + 1:min(i + window_size + 1, len(sentence))]
            # 如果长度不够，用"<pad>"补齐
            while len(context_words) < 2 * window_size:
                context_words.append("<pad>")
            data.append((word, context_words))
    return data

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear_output = nn.Linear(embedding_dim, vocab_size, bias=False)
    
    def forward(self, x):
        x = self.embedding(x)  # 获取embedding
        x = torch.mean(x, dim=1)  # 对上下文词向量取平均
        x = self.linear_output(x)  # 输出层
        return x

def train(model, dataloader, device, language, val_dataloader, epochs=30, lr=0.005):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    best_acc = 0
    patience = 3

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = loss_fn(output, y)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        # 验证模型
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for X, y in val_dataloader:
                X, y = X.to(device), y.to(device)
                output = model(X)
                _, predicted = torch.max(output, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        acc = 100 * correct / total
        best_acc = max(best_acc, acc)
        print("Epoch {}: Loss: {:.4f}, Validation accuracy: {:.2f}%".format(epoch + 1, total_loss / len(dataloader), acc))

        # 保存模型在./models文件夹下
        if not os.path.exists("models"):
            os.makedirs("models")
        torch.save(model.state_dict(), f"models/cbow_{language}_{epoch}_{lr}_256.pth")

        # 利用validation set来early stop
        if epoch > 1:
            if acc < best_acc:
                patience -= 1
            else:
                patience = 3
            if patience == 0:
                print("Early stopping...")
                break

    # 保存最终模型
    torch.save(model.state_dict(), f"cbow_{language}.pth")

def test(model, dataloader, device, language):
    model.load_state_dict(torch.load(f"cbow_{language}.pth", weights_only=True))
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            _, predicted = torch.max(output, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        print("Accuracy on {} dataset: {:.2f}%".format(language, 100 * correct / total))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    # 固定seed
    torch.manual_seed(2024)
    np.random.seed(2024)

    # 读取词汇表
    print("Reading vocab...")
    jpn_word2id, jpn_id2word, jpn_size = read_vocab("jpn_vocab.txt")
    eng_word2id, eng_id2word, eng_size = read_vocab("eng_vocab.txt")

    # 读取数据
    print("Reading sentences...")
    jpn_sentences, eng_sentences = read_sentences("train.txt")
    jpn_val_sentences, eng_val_sentences = read_sentences("val.txt")
    jpn_test_sentences, eng_test_sentences = read_sentences("test.txt")

    # 构建数据
    print("Building data...")
    jpn_data = build_jpn_data(jpn_sentences, jpn_word2id)
    eng_data = build_eng_data(eng_sentences, eng_word2id)
    jpn_val_data = build_jpn_data(jpn_val_sentences, jpn_word2id)
    eng_val_data = build_eng_data(eng_val_sentences, eng_word2id)
    jpn_test_data = build_jpn_data(jpn_test_sentences, jpn_word2id)
    eng_test_data = build_eng_data(eng_test_sentences, eng_word2id)

    # 创建数据集和数据加载器
    batch_size = 256
    jpn_dataset = CBOWDataset(jpn_data, jpn_word2id)
    jpn_dataloader = DataLoader(jpn_dataset, batch_size=batch_size, shuffle=True)
    jpn_val_dataset = CBOWDataset(jpn_val_data, jpn_word2id)
    jpn_val_dataloader = DataLoader(jpn_val_dataset, batch_size=batch_size, shuffle=True)
    jpn_test_dataset = CBOWDataset(jpn_test_data, jpn_word2id)
    jpn_test_dataloader = DataLoader(jpn_test_dataset, batch_size=batch_size, shuffle=True)

    eng_dataset = CBOWDataset(eng_data, eng_word2id)
    eng_dataloader = DataLoader(eng_dataset, batch_size=batch_size, shuffle=True)
    eng_val_dataset = CBOWDataset(eng_val_data, eng_word2id)
    eng_val_dataloader = DataLoader(eng_val_dataset, batch_size=batch_size, shuffle=True)
    eng_test_dataset = CBOWDataset(eng_test_data, eng_word2id)
    eng_test_dataloader = DataLoader(eng_test_dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型
    jp_model = CBOW(jpn_size).to(device)
    en_model = CBOW(eng_size).to(device)

    # 训练模型
    print("Training English model...")
    train(en_model, eng_dataloader, device, "eng", eng_val_dataloader)
    print("Training Japanese model...")
    train(jp_model, jpn_dataloader, device, "jpn", jpn_val_dataloader)
    print("Done")

    # 测试模型
    print("Testing Japanese model...")
    test(jp_model, jpn_test_dataloader, device, "jpn")
    print("Testing English model...")
    test(en_model, eng_test_dataloader, device, "eng")
