import os
import MeCab
import nltk

# 构建日语词汇表
def build_jpn_vocab():
    # 读取日语数据
    with open("eng_jpn.txt", "r", encoding="utf-8") as f:
        data = f.readlines()
        jpn = [line.split("\t")[0] for line in data]
    # 使用MeCab分词
    m = MeCab.Tagger("-Owakati")
    jpn = [m.parse(sentence).split() for sentence in jpn]
    
    # 构建词汇表
    vocab = set()
    for word in jpn:
        vocab.update(word)
    vocab = list(vocab)
    vocab.sort()
    vocab = ["<pad>", "<unk>", "<sos>", "<eos>"] + vocab
    with open("jpn_vocab.txt", "w", encoding="utf-8") as f:
        for word in vocab:
            f.write(word + "\n")
    print("Done")

def eng_tokenize(sentence):
    words = []
    for word in sentence.split():
        word = word.strip(",.!?\"")
        word = word.lower()
        words.append(word)
    return words

# 构建英语词汇表
def build_eng_vocab():
    # 读取英语数据
    with open("eng_jpn.txt", "r", encoding="utf-8") as f:
        data = f.readlines()
        eng_sentence = [line.split("\t")[1]for line in data]
    eng_words = [nltk.word_tokenize(sentence) for sentence in eng_sentence]
    
    # 构建词汇表
    vocab = set()
    for word in eng_words:
        # 将标点提取出来
        word = [w.strip(",.!?\"") for w in word]
        
        vocab.update(word)
    vocab = list(vocab)
    vocab.sort()
    vocab = [",", ".", "!", "?", "\""] + vocab
    vocab = ["<pad>", "<unk>", "<sos>", "<eos>"] + vocab
    with open("eng_vocab.txt", "w", encoding="utf-8") as f:
        for word in vocab:
            f.write(word + "\n")
    print("Done")


if __name__ == "__main__":
    build_jpn_vocab()
    build_eng_vocab()