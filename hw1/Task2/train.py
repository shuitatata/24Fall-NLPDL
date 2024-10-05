import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from utils import clean_non_chinese_chars, read_vocab, load_dataset
from cnn import CNN_Text
import jieba

# 定义超参数
EPOCHS = 10        # 训练轮次
BATCH_SIZE = 16    # 每个批次的数据量
LEARNING_RATE = 0.001  # 学习率
EMBED_DIM = 256    # 词向量维度
CLASS_NUM = 4      # 分类数目，二分类为2
KERNEL_SIZES = [2, 3, 4, 5]  # 卷积核大小
NUM_FILTERS = 128  # 每个卷积核的输出通道数
DROPOUT = 0.4      # Dropout 概率
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 使用GPU或CPU
SEED = 20040618

def train_model(model, train_loader,val_loader, optimizer, criterion, device, num_epochs=10):
    model.train()
    best_val_acc = 0
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            # texts = texts.unsqueeze(2)
            outputs = model(texts)
            # print(outputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            # print(predicted, labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        val_total = 0
        val_correct = 0

        model.eval()
        with torch.no_grad():
            for texts, labels in val_loader:
                texts, labels = texts.to(device), labels.to(device)
                outputs = model(texts)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_acc = val_correct / val_total
        avg_loss = total_loss / len(train_loader)
        acc = correct / total

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Acc: {acc:.4f}, Val_Acc: {val_acc:.4f}')
        # save model

        # Check for early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            # Save the best model
            torch.save(model.state_dict(), f'checkpoints/cnn_text_best.pt')
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= 2:
            print(f'Early stopping at epoch {epoch+1}')
            break

        torch.save(model.state_dict(), f'checkpoints/cnn_text_epoch_{epoch}.pt')

def test_model(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        acc = correct / total
        print(f'Test Accuracy: {acc:.4f}')


if __name__ == '__main__':
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    jieba.load_userdict('vocabulary.txt')
    vocab = read_vocab('vocabulary.txt')

    # 填充词的索引为0， 不在词表中的词索引为1
    num_vocab = len(vocab) + 2

    dataset = load_dataset('train.txt', vocab)
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    test_dataset = load_dataset('test.txt', vocab)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = CNN_Text(embed_dim=EMBED_DIM,num_vocab=num_vocab, class_num=CLASS_NUM, kernel_sizes=KERNEL_SIZES, num_filters=NUM_FILTERS, dropout=DROPOUT)
    model.to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_model(model, train_loader,val_loader, optimizer, criterion, DEVICE, num_epochs=EPOCHS)

    test_model(model, test_loader, DEVICE)


    

    