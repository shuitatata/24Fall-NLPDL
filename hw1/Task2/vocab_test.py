# 测试词汇表的命中率

import jieba
from vocab_construction import clean_non_chinese_chars

vocab = []
misswords = []
with open('vocabulary.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        vocab.append(line.strip())

with open('test.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    total_words = 0
    hit_words = 0
    for line in lines:
        sentence = line.strip().split('\t')[0]
        sentence = clean_non_chinese_chars(sentence)
        words = jieba.lcut(sentence, cut_all=False)
        total_words += len(words)
        for word in words:
            if word in vocab:
                hit_words += 1
            else:
                misswords.append(word)

print('Total words:', total_words)
print('Hit words:', hit_words)
print('Miss words:', misswords[:100])