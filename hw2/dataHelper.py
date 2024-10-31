from datasets import load_dataset, Dataset, DatasetDict
import json

def load_SemEval14(path, sep_token):

    polarity_map = {"positive": 3, "neutral": 2, "negative": 1, "conflict": 0}

    with open (path + "/train.json", 'r') as f:
        data = json.load(f)
    train_data = [{"text": data[id]["term"]+sep_token+data[id]["sentence"], "label": polarity_map[data[id]["polarity"]]} for id in data]
    
    with open (path + "/test.json", 'r') as f:
        data = json.load(f)
    test_data = [{"text": data[id]["term"]+sep_token+data[id]["sentence"], "label": polarity_map[data[id]["polarity"]]} for id in data]
    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)
    return train_dataset, test_dataset
        

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

    print(dataset['train'][0])
    # print(type(dataset))

    return dataset

if __name__ == '__main__':
    dataset = get_dataset("laptop_sup", "[SEP]")
    print(dataset["train"][0])