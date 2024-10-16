import torch
import torch.nn as nn
from cbow import CBOW

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, pre_trained_embedding=None):
        super(Encoder, self).__init__()
        # 使用预训练的嵌入
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pre_trained_embedding is not None:
            self.embedding.weight = nn.Parameter(pre_trained_embedding)
            self.embedding.weight.requires_grad = True
        # LSTM编码器
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True)

    def forward(self, src):
        # 输入：词ID序列 -> 嵌入向量
        embedded = self.embedding(src)  # [src_len, batch_size, embedding_dim]

        # LSTM编码器：输出hidden states和cell states
        outputs, (hidden, cell) = self.lstm(embedded)  # outputs: [src_len, batch_size, hidden_dim*2]

        return outputs, hidden, cell

class DotProductAttention(nn.Module):
    def __init__(self):
        super(DotProductAttention, self).__init__()

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch_size, hidden_dim] (decoder的当前隐藏状态)
        # encoder_outputs: [src_len, batch_size, hidden_dim * num_directions] (encoder的所有时间步输出)

        hidden = torch.cat([hidden, hidden], dim=1) # [batch_size, hidden_dim * 2]

        hidden = hidden.unsqueeze(1)  # [batch_size, 1, hidden_dim * 2]

        # 点积
        # encoder_outputs.permute: [batch_size, hidden_dim * num_directions, src_len]
        attn_energies = torch.bmm(hidden, encoder_outputs.permute(1, 2, 0))  # [batch_size, 1, src_len]
        attn_energies = attn_energies.squeeze(1)  # [batch_size, src_len]

        # 2. 归一化权重 (Softmax)
        attention_weights = torch.softmax(attn_energies, dim=1)  # [batch_size, src_len]

        # print("attention_weights", attention_weights.shape)
        return attention_weights

class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, attention, num_layers=1, pre_trained_embedding=None):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        if pre_trained_embedding is not None:
            self.embedding.weight = nn.Parameter(pre_trained_embedding)
            self.embedding.weight.requires_grad = True

        self.lstm = nn.LSTM(embedding_dim + hidden_dim*2, hidden_dim, num_layers=num_layers)
        self.attention = attention
        self.fc_out = nn.Linear(hidden_dim * 3, output_dim)

    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(0)  # [1, batch_size]
        embedded = self.embedding(input)  # [1, batch_size, embedding_dim]
        # print(embedded)

        # 使用点乘Attention计算上下文向量
        attn_weights = self.attention(hidden[-1], encoder_outputs)  # [batch_size, src_len]
        attn_weights = attn_weights.unsqueeze(1)  # [batch_size, 1, src_len]

        # 上下文向量：对encoder outputs加权求和
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # [batch_size, src_len, hidden_dim * 2]
        context = torch.bmm(attn_weights, encoder_outputs)  # [batch_size, 1, hidden_dim * 2
        context = context.permute(1, 0, 2)  # [1, batch_size, hidden_dim * 2]

        # print("context", context.shape)

        # LSTM输入
        rnn_input = torch.cat((embedded, context), dim=2)  # [1, batch_size, embedding_dim + hidden_dim * 2]
        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))  # output: [1, batch_size, hidden_dim]

        # print("output", output.shape)

        # 输出预测
        output = torch.cat((output.squeeze(0), context.squeeze(0)), dim=1)  # [batch_size, hidden_dim * 3]
        prediction = self.fc_out(output)  # [batch_size, output_dim]
        # prediction = nn.Softmax(dim=1)(prediction)

        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.fc_hidden = nn.Linear(self.encoder.lstm.hidden_size * 2, self.decoder.lstm.hidden_size)
        self.fc_cell = nn.Linear(self.encoder.lstm.hidden_size * 2, self.decoder.lstm.hidden_size)
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        if trg is None:
            # 推理模式 直到输出<eos>为止
            input = src[0, :]  # 取<sos>
            # print(input)
            encoder_outputs, hidden, cell = self.encoder(src)
            outputs = []
            output_dim = self.decoder.fc_out.out_features
            num_layers = hidden.shape[0] // 2
            batch_size = hidden.shape[1]
            hidden_dim = hidden.shape[2]
            hidden = hidden.view(num_layers, 2, batch_size, hidden_dim)
            hidden = hidden.permute(0, 2, 1, 3)
            hidden = hidden.reshape(num_layers, batch_size, hidden_dim * 2)

            hidden = self.fc_hidden(hidden)

            cell = cell.view(num_layers, 2, batch_size, hidden_dim)
            cell = cell.permute(0, 2, 1, 3)
            cell = cell.reshape(num_layers, batch_size, hidden_dim * 2)

            cell = self.fc_cell(cell)
            # batch_size = 1

            for t in range(1, 100):
                output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
                
                top1 = output.argmax(1)
                
                outputs.append( top1)
                input = top1
                if top1.item() == 3:  # <eos>
                    break

            return outputs

        
        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        output_dim = self.decoder.fc_out.out_features

        # 保存解码器输出
        outputs = torch.zeros(trg_len, batch_size, output_dim).to(self.device)

        # 编码器前向传播
        encoder_outputs, hidden, cell = self.encoder(src)

        num_layers = hidden.shape[0] // 2
        batch_size = hidden.shape[1]
        hidden_dim = hidden.shape[2]
        hidden = hidden.view(num_layers, 2, batch_size, hidden_dim)
        hidden = hidden.permute(0, 2, 1, 3)
        hidden = hidden.reshape(num_layers, batch_size, hidden_dim * 2)

        hidden = self.fc_hidden(hidden)

        cell = cell.view(num_layers, 2, batch_size, hidden_dim)
        cell = cell.permute(0, 2, 1, 3)
        cell = cell.reshape(num_layers, batch_size, hidden_dim * 2)

        cell = self.fc_cell(cell)

        # 初始输入为<sos> token
        input = trg[0, :]
        # debug = []
        for t in range(1, trg_len):
            # 解码器前向传播
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[t] = output

            # Teacher Forcing机制
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)  # 获取最大概率的单词
            # debug.append(top1)
            input = trg[t] if teacher_force else top1
            
        # print("top1", debug)
        return outputs
