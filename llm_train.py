import os
from os import listdir 
from os.path import isfile, join

import argparse
import torch
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset

class TrainerDataset(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path, sep="\t")
        df = df.dropna()
        self.dataset = df
        self.tokenizer = T5Tokenizer.from_pretrained(hf_model)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        source = self.dataset.iloc[idx, 0]
        target = self.dataset.iloc[idx, 1]
        input_ids = self.tokenizer.encode(source, return_tensors='pt',
                                          padding='max_length', truncation='longest_first', max_length=512)[0]
        label = self.tokenizer.encode(target, return_tensors='pt', padding='max_length',
                                      truncation='longest_first', max_length=64)[0]
        return {'input_ids': input_ids, 'labels': label}



'''
Train doc2query on MS MARCO with t5 from hgf

The training data should contains source and target in each line, which should be separate by '\t'
'''

hf_model = 'ai-forever/ruT5-large'




data_path = os.path.join(os.getcwd(), "datasets")
datasets_file_names = [f for f in listdir(data_path) if isfile(join(data_path, f))]

for fn in datasets_file_names:
    train_data_path = os.path.join(os.getcwd(), "datasets", fn)
    parser = argparse.ArgumentParser(description='Train docTquery on more datasets')
    learned_model_path = os.path.join(os.getcwd(), "great_validator")
    parser.add_argument('--pretrained_model_path', default=hf_model, help='pretrained model path')
    parser.add_argument('--train_data_path', default=train_data_path, required=False, help='training data path')
    parser.add_argument('--output_path', default=learned_model_path, required=False, help='output directory path')
    parser.add_argument('--epoch', default=1, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=5e-5, type=float)
    parser.add_argument('--lr', default=3e-5, type=float)
    parser.add_argument('--gra_acc_steps', default=8, type=int)
    # parser.add_argument('--vocab_size', default=40000, type=int)
    args = parser.parse_args()

    model = T5ForConditionalGeneration.from_pretrained(args.pretrained_model_path).to('cuda')
    train_dataset = TrainerDataset(args.train_data_path)

    training_args = TrainingArguments(
        output_dir=args.output_path,
        overwrite_output_dir=True,
        num_train_epochs=args.epoch,
        per_device_train_batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        save_steps=1000,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.gra_acc_steps,
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )

    trainer.train()
    trainer.save_model()