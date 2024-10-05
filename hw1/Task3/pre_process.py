# read the eng_jpn.txt and split it into 3 set
import os
import random

def read_data():
    with open("eng_jpn.txt", "r", encoding="utf-8") as f:
        data = f.readlines()
    return data

def split_data(data):
    random.shuffle(data)
    train_data = data[:int(len(data)*0.8)]
    test_data = data[int(len(data)*0.8):int(len(data)*0.9)]
    val_data = data[int(len(data)*0.9):]
    return train_data, test_data, val_data

def write_data(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for line in data:
            f.write(line)

if __name__ == "__main__":
    data = read_data()
    train_data, test_data, val_data = split_data(data)
    write_data(train_data, "train.txt")
    write_data(test_data, "test.txt")
    write_data(val_data, "val.txt")
    print("Done")