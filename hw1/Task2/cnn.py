import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Text(nn.Module):
    def __init__(self, num_vocab, embed_dim, class_num, kernel_sizes=[3, 4, 5], num_filters=100, dropout=0.5):
        super(CNN_Text, self).__init__()
        
        self.embedding = nn.Embedding(num_vocab, embed_dim)

        # 卷积层：多个不同大小的卷积核，提取不同n-gram特征
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (k, embed_dim)) for k in kernel_sizes]
        )
        
        # Dropout：正则化层
        self.dropout = nn.Dropout(dropout)
        
        # 全连接层：将池化后的特征映射到类别数
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, class_num)

        self.softmax = nn.Softmax(dim=1)

    def conv_and_pool(self, x, conv):
        # 进行卷积和最大池化操作
        x = F.relu(conv(x)).squeeze(3)  # Conv2d 的输出是四维的，squeeze去除冗余维度
        x = F.max_pool1d(x, x.size(2)).squeeze(2)  # 最大池化
        return x

    def forward(self, x):
        # 假设 x: [batch_size, sentence_length] 输入为未嵌入的向量
        
        x = self.embedding(x) #[batch_size, sentence_length, embed_dim]
        x = x.unsqueeze(1) # [batch_size, 1, sentence_length, embed_dim]

        # 对每个卷积层进行卷积和池化，并拼接结果
        x = torch.cat([self.conv_and_pool(x, conv) for conv in self.convs], 1)

        # Dropout
        x = self.dropout(x)

        # 全连接层
        x = self.fc(x)

        # softmax
        x = self.softmax(x)

        return x


