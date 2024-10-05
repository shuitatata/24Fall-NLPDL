import re
import jieba
import numpy as np

def clean_non_chinese_chars(text):
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    clean_text = re.sub(pattern, '', text)
    return clean_text

# pad a sequence to the given length
def pad_sequence(sequence, length=18, pad_idx=0):
    padded_sequence = np.zeros(length, dtype=np.int64)
    # 打印sequence的长度
    # print(len(sequence))
    padded_sequence[:len(sequence)] = sequence
    return padded_sequence

# load and index the vocabulary
def read_vocab(vocab_file):
    vocab = {}
    with open(vocab_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            word = line.strip()
            vocab[word] = idx+2
    return vocab

# load the dataset
def load_dataset(file_path, vocab):
    dataset = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            sentence = line.strip().split('\t')[0]
            label = line.strip().split('\t')[1]
            sentence = clean_non_chinese_chars(sentence)
            words = jieba.lcut(sentence, cut_all=False)

            # if word not in vocab, use -1 to represent the word
            word_indices = [vocab.get(word, 1) for word in words]
            word_indices = pad_sequence(word_indices)
            # convert word_indices to float
            # word_indices = np.array(word_indices, dtype=np.float32)

            dataset.append((word_indices, int(label)))
    return dataset

if __name__ == '__main__':
    vocab = read_vocab('vocabulary.txt')
    train_dataset = load_dataset('train.txt', vocab)

    max_len = 0
    for text, label in train_dataset:
        print(text)
        max_len = max(max_len, len(text))
    print('Max sequence length:', max_len)
