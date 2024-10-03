import jieba
import re

jieba.load_userdict('vocabulary.txt')
def clean_non_chinese_chars(text):
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    clean_text = re.sub(pattern, '', text)
    return clean_text

words = []

with open('dev.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        sentence = line.strip().split('\t')[0]
        sentence = clean_non_chinese_chars(sentence)
        words.extend(jieba.lcut(sentence, cut_all=False))
    
with open('train.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        sentence = line.strip().split('\t')[0]
        sentence = clean_non_chinese_chars(sentence)
        words.extend(jieba.lcut(sentence, cut_all=False))

words = list(set(words))

with open('vocabulary.txt', 'w', encoding='utf-8') as f:
    for word in words:
        f.write(word + '\n')
        