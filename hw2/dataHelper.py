from datasets import load_dataset, Dataset, DatasetDict
import json
import jsonlines
import pandas as pd

def load_SemEval14(path, sep_token):

    polarity_map = {"positive": 3, "neutral": 2, "negative": 1, "conflict": 0}

    with open (path + "/train.json", 'r') as f:
        data = json.load(f)
    train_data = [{"text": data[id]["term"]+sep_token+data[id]["sentence"], "labels": polarity_map[data[id]["polarity"]]} for id in data]
    
    with open (path + "/test.json", 'r') as f:
        data = json.load(f)
    test_data = [{"text": data[id]["term"]+sep_token+data[id]["sentence"], "labels": polarity_map[data[id]["polarity"]]} for id in data]
    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)
    return train_dataset, test_dataset
        
def load_aclarc(path, sep_token):
    data = {}
    train_data = []
    test_data= []
    label_map = {'Uses': 0, 'Future': 1, 'CompareOrContrast': 2,
                     'Motivation': 3, 'Extends': 4, 'Background': 5}
    with open(path+"/train.jsonl", 'r') as f:
        raw_data = jsonlines.Reader(f)
        for item in raw_data:
            train_data += [{"text": item["text"], "labels": label_map[item["label"]]}]

    with open(path+"/test.jsonl", 'r') as f:
        raw_data = jsonlines.Reader(f)
        for item in raw_data:
            test_data += [{"text": item["text"], "labels": label_map[item["label"]]}]

    return DatasetDict({"train": Dataset.from_list(train_data), "test": Dataset.from_list(test_data)})

def get_dataset(dataset_name, sep_token):
    '''
    dataset_name: str, the name of the dataset
    sep_token: str, the sep_token used by tokenize
    '''
    dataset = None

    if dataset_name == "restaurant_sup":
        train_dataset, test_dataset = load_SemEval14("data/SemEval14-res", sep_token)
        dataset = DatasetDict({"train": train_dataset, "test": test_dataset})
        return dataset

    if dataset_name == "laptop_sup":
        train_dataset, test_dataset = load_SemEval14("data/SemEval14-laptop", sep_token)
        dataset = DatasetDict({"train": train_dataset, "test": test_dataset})
        return dataset

    if dataset_name == "acl_sup":
        dataset = load_aclarc("data/acl_sup", sep_token)
        return dataset

    if dataset_name == "agnews_sup":
        # dataset = load_dataset("csv", data_files="data/agnews_sup/agnews_sup.csv")
        df = pd.read_csv('data/agnews_sup/agnews_sup.csv', header=0)
        df['labels'] = df['labels'] - 1
        dataset = Dataset.from_pandas(df)
        # print(dataset)
        dataset = dataset.train_test_split(test_size=0.1, seed=2022, shuffle=True)
        return dataset

    if "fs" in dataset_name:
        dataset = get_dataset(dataset_name[:-2]+"sup", sep_token)
        dataset["train"] = dataset["train"].select(range(32))
        # dataset = DatasetDict({"train": dataset["train"]
        # [:32], "test": dataset["test"]})
        return dataset

    # dataset_name是列表
    if isinstance(dataset_name, list):
        all_train = []
        all_test = []
        all_label = set()
        for name in dataset_name:
            dataset = get_dataset(name, sep_token)
            label_offset = len(all_label) 
            dataset['train'] = dataset['train'].map(lambda x: {'text': x['text'], 'labels': x['labels']+label_offset})
            dataset['test'] = dataset['test'].map(lambda x: {'text': x['text'], 'labels': x['labels']+label_offset})
            all_train += dataset["train"]
            all_test += dataset["test"]
            all_label = all_label.union(set(dataset["train"]["labels"]))
        dataset = DatasetDict({"train": Dataset.from_list(all_train), "test": Dataset.from_list(all_test)})
        print(all_label)


    # print(type(dataset))

    return dataset

if __name__ == '__main__':
    # dataset = get_dataset("restaurant_sup", "[SEP]")
    # print(dataset['train'][0])
    # print(load_aclarc("data/acl_sup", "[SEP]"))
    print(get_dataset("agnews_sup", "[SEP]"))
    print(get_dataset("acl_sup", "[SEP]"))
    print(get_dataset(["agnews_sup", "acl_sup"], "[SEP]"))