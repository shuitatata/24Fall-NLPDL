import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import evaluate
import numpy as numpy

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry
from dataHelper import get_dataset

import wandb

os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Initialize logging, random seed, parameters and wandb
@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

@dataclass
class DataArguments:
    dataset_name: str = field(
        default="agnews_sup",
        metadata={"help": "Name of the dataset to use"}
    )
    max_seq_length: int = field(
        default=128,
        metadata={"help": "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."}
    )
    sep_token: str = field(
        default="[SEP]",
        metadata={"help": "The sep token used by the tokenizer"}
    )

@dataclass
class MyTrainingArguments(TrainingArguments):
    output_dir: str = "./output"
    do_train: bool = True
    do_eval: bool = True
    eval_strategy: str = "epoch"
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    num_train_epochs: int = 3
    weight_decay: float = 0.01
    report_to: str = "wandb"
    logging_steps: int = 10
    save_steps: int = 0
    save_total_limit: int = 0
    seed: int

parser = HfArgumentParser((ModelArguments, DataArguments, MyTrainingArguments))
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
else:
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers = [logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
# seed=23534
set_seed(training_args.seed)

run_name=f"{model_args.model_name_or_path}-{data_args.dataset_name}-{training_args.seed}"
wandb.init(project="nlp-transformers", name=run_name)
wandb.config.update({**vars(model_args), **vars(data_args), **vars(training_args)})
training_args.output_dir = os.path.join(training_args.output_dir, run_name)

# Load the dataset
logger.info(f"Loading dataset {data_args.dataset_name}")
raw_dataset = get_dataset(data_args.dataset_name, data_args.sep_token)

# Load the model and tokenizer
logger.info(f"Loading model {model_args.model_name_or_path}")
config = AutoConfig.from_pretrained(model_args.model_name_or_path, num_labels=6)
tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path, config=config)



# Preprocess the dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        padding="max_length", 
        truncation=True,
        max_length=data_args.max_seq_length,
    )

tokenized_datasets = raw_dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer)

# Define the compute_metrics function
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(p: EvalPrediction):
    predictions, labels = p
    preds = predictions.argmax(-1)
    
    accuracy = accuracy_metric.compute(predictions=preds, references=labels)
    
    micro_f1 = f1_metric.compute(predictions=preds, references=labels, average="micro")
    macro_f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")
    
    return {
        "accuracy": accuracy["accuracy"],
        "micro_f1": micro_f1,
        "macro_f1": macro_f1
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

if training_args.do_train:
    logger.info("Start training")
    trainer.train()
    trainer.save_model(training_args.output_dir)
    trainer.save_metrics("train", trainer.evaluate())

if training_args.do_eval:
    metrics = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
    trainer.save_metrics("eval", metrics)

wandb.finish()

# # preview the dataset
# logger.info("Preview of the first training example")
# logger.info(tokenized_datasets["train"][0])
