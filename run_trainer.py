import os
import sys
import numpy as np
import pandas as pd
import json
import math
import random
import re
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Classifier")

from tqdm import tqdm
import sklearn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from transformers.hf_argparser import HfArgumentParser

from utils.classifier import ModelDefine, DataManager, Arguments

class Trainer:
    def __init__(self, args, device):
        if device is None:
            self.device = 'cuda'
        else:
            self.device = device
        self.dt = DataManager(args)
        self.args = args
        if 'pkl' in self.args.model_name_or_path:
            self.model = torch.load(self.args.model_name_or_path)
            logger.info(f'load best model from {self.args.model_name_or_path}')
        else:
            self.model = ModelDefine(args)
        self.model.to(self.device)
        self.train_loss = AverageMeter()
        self.updates = 0
        if not os.path.exists(self.args.model_save_dir):
            os.makedirs(self.args.model_save_dir, exist_ok=True)
        self.optimizer = optim.Adamax([p for p in self.model.parameters() if p.requires_grad], lr=args.learning_rate)
        self.best_acc = 0.
        
    def train(self):
        logger.info("=" * 50)
        for i in range(self.args.epochs):
            logger.info("epoch: %d" % i)
            with tqdm(enumerate(self.dt.iter_batches(which="train", batch_size=self.args.train_batch_size)), ncols=80) as t:
                for batch_id, batch in t:
                    self.model.train()
                    input_ids, token_type_ids, attention_mask, labels = [
                        Variable(e).long().to(self.device) for e
                        in batch]
                    labels = labels.clone().detach().unsqueeze(0)
                    _, loss = self.model(
                        input_ids=input_ids,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        do_train=True,) 
                    self.train_loss.update(loss.item(), self.args.train_batch_size)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    t.set_postfix(loss=self.train_loss.avg)
                    t.update(1)
                    self.updates += 1

            logger.info("epoch {}, train loss={}".format(i, self.train_loss.avg))
            self.train_loss.reset()
            acc = self.validate(epoch=i, which='dev')
            if acc > self.best_acc:
                self.best_acc = acc
                save_path = os.path.join(self.args.model_save_dir, 'checkpoint_best.pkl')
                logger.info(f"save model to {save_path}")
                torch.save(self.model, save_path)

    def validate(self, which="test", epoch=-1):
        def calculate_acc(labels, scores):
            y_pred = np.argmax(scores, axis=1)
            correct_predictions = np.sum(y_pred == labels)
            accuracy = correct_predictions / len(labels)
            return accuracy
        
        gold_label = []
        y_predprob = []
        for batch in tqdm(self.dt.iter_batches(which=which, batch_size=self.args.eval_batch_size), ncols=80):
            self.model.eval()
            input_ids, token_type_ids, attention_mask, labels = [
                    Variable(e).long().to(self.device) for e
                    in batch]

            labels = labels.clone().detach().unsqueeze(0)
            logits = self.model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=labels,
                do_train=False,)

            gold_label.extend(batch[-1])
            y_predprob.extend(F.softmax(logits, dim=1).detach().cpu().numpy())
        gold_label = np.array(gold_label)
        y_predprob = np.array(y_predprob)
        acc = calculate_acc(gold_label, y_predprob)
        logger.info(f"{which} acc={acc}")
        return acc

class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    parser = HfArgumentParser(Arguments)
    args: Arguments
    # if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    #     args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))[0]
    # else:
    #     args = parser.parse_args_into_dataclasses()[0]
    # test
    args = parser.parse_json_file(json_file="/raid/shnu/wjs/ftembedding/configs/train_config.json")[0]
    logger.info(args)
    setup_seed(0)
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'gpu'
    trainer = Trainer(
        args=args, 
        device=device
        )
    if args.do_train:
        trainer.train()
    elif args.do_test:
        trainer.validate(epoch=0, which='test')
