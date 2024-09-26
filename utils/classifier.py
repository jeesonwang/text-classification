import os
import math
import json
import pathlib
from typing import Optional, List, Tuple, Dict, Any, Union, Callable
from dataclasses import dataclass, field
from collections import Counter
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import torch
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from transformers import DebertaV2Model, DebertaV2Tokenizer, AutoModel, AutoTokenizer
from torch.nn import CrossEntropyLoss, BCELoss
from torch.utils.data import Dataset, DataLoader

from .import_utils import import_custom_func
from .act_fn import ACT2FN
from .classifier_loss import FocalLoss

@dataclass
class Arguments:
    """
    Arguments for training model.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the model."},
    )
    tokenizer_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the tokenizer."},
    )
    model_save_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save the finetuned model."},
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": ""}
    )
    train_batch_size: int = field(
        default=None,
        metadata={"help": ""}
    )
    eval_batch_size: int = field(
        default=None,
        metadata={"help": ""}
    )
    epochs: int = field(
        default=None,
        metadata={"help": ""}
    )
    do_train: bool = field(
        default=None,
        metadata={"help": ""}
    )
    do_test: bool = field(
        default=None,
        metadata={"help": ""}
    )
    load_checkpoints: Optional[str] = field(
        default=None,
        metadata={"help": "Path to load the fine-tuned checkpoint."},
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the train set"},
    )
    dev_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the dev set"},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the test set"},
    )
    output_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save result"},
    )
    max_train_samples: int = field(
        default=None,
        metadata={"help": "max train samples"}
    )
    preprocess_fn_path: str = field(
        default="utils/utils.py",
        metadata={"help": "Path to the preprocess function"}
    )
    preprocess_fn_name: str = field(
        default="preprocess_content",
        metadata={"help": "Name of the preprocess function"}
    )
    num_labels: int = field(
        default=None,
        metadata={"help": "Number of labels"}
    )

class DataManager:
    def __init__(self, args: Arguments):
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        self.tokenizer.model_max_length = 512
        self.args = args
        self.preprocess_fn = import_custom_func(args.preprocess_fn_path, args.preprocess_fn_name)
        self.label_process()
        args.num_labels = len(self.label_to_id)
        self.prepare_data()

    def prepare_data(self):
        if self.args.do_train:
            train_smps = self.get_data_2s('train')
            dev_smps = self.get_data_2s('dev')
            self.train_dataset = MyDataSet(train_smps)
            self.dev_dataset = MyDataSet(dev_smps)
        if self.args.do_test:
            test_smps = self.get_data_2s('test')
            self.test_dataset = MyDataSet(test_smps)

    @staticmethod
    def data_load(file_path: str) -> Callable:
        data_path = pathlib.Path(file_path)
        if data_path.suffix == '.csv':
            data_load_fn = lambda x: pd.read_csv(x)
        elif data_path.suffix == '.xlsx':
            data_load_fn = lambda x: pd.read_excel(x)
        else:
            raise ValueError(f'Unknown data format: {data_path.suffix}')

        data = data_load_fn(file_path)

        if "content" not in data.columns or "label" not in data.columns:
            logger.error(f'No content or label column in {file_path}, your input data should have content and label columns')
            raise ValueError
        return data

    def get_data_2s(self, which):

        if which == 'train':
            data = self.data_load(self.args.train_file)
        elif which == 'dev':
            data = self.data_load(self.args.dev_file)
        elif which == 'test':
            data = self.data_load(self.args.test_file)

        smps = []
        for idx, item in tqdm(data.iterrows(), total=len(data)):
            if which == "train" and self.args.max_train_samples and idx >= self.args.max_train_samples:
                break
            input_str = self.preprocess_fn(item['content'], which="content")
            input = self.tokenizer(input_str, padding="max_length", truncation=True, return_tensors="pt")
            label = torch.tensor(self.label_to_id[self.preprocess_fn(item['label'], which="label")], dtype=torch.long)

            smps.append([input["input_ids"][0], input["token_type_ids"][0], input["attention_mask"][0], label])
            
        logger.info(f'{which} sample num={len(smps)}')
        return smps

    def iter_batches(self, which="train", samples=None, batch_size=None):
        if which == 'train':
            return DataLoader(shuffle=False, dataset=self.train_dataset, batch_size=batch_size)
        elif which == 'dev':
            return DataLoader(shuffle=False, dataset=self.dev_dataset, batch_size=batch_size)
        elif which == 'test':
            return DataLoader(shuffle=False, dataset=self.test_dataset, batch_size=batch_size)

    def label_process(self):
        if os.path.exists(os.path.join(self.args.model_save_dir, 'label_info.json')):
            with open(os.path.join(self.args.model_save_dir, 'label_info.json'), 'r') as f:
                label_info = json.load(f)
            self.label_to_id = label_info['label_to_id']
            self.id_to_label = label_info['id_to_label']
        else:
            labels = self.data_load(self.args.train_file)['label']
            if labels is None:
                raise ValueError("label_info.json file not found and labels is None")
            if isinstance(labels, pd.Series):
                labels = labels.tolist()
            labels = list(set(labels))
            self.label_to_id = {self.preprocess_fn(label, which="label"): idx for idx, label in enumerate(labels)}
            self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
            save_path = os.path.join(self.args.model_save_dir, 'label_info.json')
            with open(save_path, 'w') as f:
                json.dump({'label_to_id': self.label_to_id, 'id_to_label': self.id_to_label}, f, indent=4, ensure_ascii=False)

class MyDataSet(Dataset):
    def __init__(self, smps):
        self.smps = smps
        super().__init__()

    def __getitem__(self, i):
        input_ids, token_type_ids, attention_mask, label= self.smps[i]
        return input_ids, token_type_ids, attention_mask, label
    
    def __len__(self):
        return len(self.smps)

class DropoutContext(object):
    def __init__(self):
        self.dropout = 0
        self.mask = None
        self.scale = 1
        self.reuse_mask = True

def get_mask(input, local_context):
    if not isinstance(local_context, DropoutContext):
        dropout = local_context
        mask = None
    else:
        dropout = local_context.dropout
        dropout *= local_context.scale
        mask = local_context.mask if local_context.reuse_mask else None

    if dropout > 0 and mask is None:
        mask = (1 - torch.empty_like(input).bernoulli_(1 - dropout)).bool()

    if isinstance(local_context, DropoutContext):
        if local_context.mask is None:
            local_context.mask = mask

    return mask, dropout

class XDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, local_ctx):
        mask, dropout = get_mask(input, local_ctx)
        ctx.scale = 1.0 / (1 - dropout)
        if dropout > 0:
            ctx.save_for_backward(mask)
            return input.masked_fill(mask, 0) * ctx.scale
        else:
            return input
    @staticmethod
    def backward(ctx, grad_output):
        if ctx.scale > 1:
            (mask,) = ctx.saved_tensors
            return grad_output.masked_fill(mask, 0) * ctx.scale, None
        else:
            return grad_output, None

# Copied from transformers.models.deberta.modeling_deberta.StableDropout
class StableDropout(nn.Module):
    """
    Optimized dropout module for stabilizing the training
    Args:
        drop_prob (float): the dropout probabilities
    """
    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob
        self.count = 0
        self.context_stack = None

    def forward(self, x):
        """
        Call the module
        Args:
            x (`torch.tensor`): The input tensor to apply dropout
        """
        if self.training and self.drop_prob > 0:
            return XDropout.apply(x, self.get_context())
        return x

    def clear_context(self):
        self.count = 0
        self.context_stack = None

    def init_context(self, reuse_mask=True, scale=1):
        if self.context_stack is None:
            self.context_stack = []
        self.count = 0
        for c in self.context_stack:
            c.reuse_mask = reuse_mask
            c.scale = scale

    def get_context(self):
        if self.context_stack is not None:
            if self.count >= len(self.context_stack):
                self.context_stack.append(DropoutContext())
            ctx = self.context_stack[self.count]
            ctx.dropout = self.drop_prob
            self.count += 1
            return ctx
        else:
            return self.drop_prob

class ContextPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.pooler_hidden_size)
        self.dropout = StableDropout(config.pooler_dropout)
        self.config = config

    def forward(self, hidden_states):
        """
        We "pool" the model by simply taking the hidden state corresponding
        to the first token.
        """
        context_token = hidden_states[:, 0]
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = ACT2FN[self.config.pooler_hidden_act](pooled_output)
        return pooled_output

    def output_dim(self):
        return self.config.hidden_size
    
class ModelDefine(nn.Module):
    def __init__(self, args):
        super(ModelDefine, self).__init__()
        logger.info(args.model_name_or_path)
        self.deberta = AutoModel.from_pretrained(args.model_name_or_path)
        self.config = self.deberta.config
        num_labels = args.num_labels if args.num_labels else getattr(self.config, "num_labels", 2)
        self.num_labels = num_labels
        self.config.num_labels = num_labels
        if not hasattr(self.config, "pooler_hidden_size"):
            self.config.pooler_hidden_size = self.config.pooler_fc_size
            self.config.pooler_dropout = self.config.hidden_dropout_prob
            self.config.pooler_hidden_act = self.config.hidden_act
        self.config.classifier_dropout = 0.0
        self.pooler = ContextPooler(self.config)
        self.classifier = nn.Linear(self.config.pooler_hidden_size, num_labels)
        drop_out = getattr(self.config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)
        self.alpha = 0.5
        # self.loss_fn = CrossEntropyLoss()
        self.loss_fn = FocalLoss()
    
    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        do_train=None,
    ):       
        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if do_train:
            loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            return logits, loss
        else:
            return logits
