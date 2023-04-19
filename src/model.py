import pdb

import pytorch_lightning as pl
import torch
from sklearn.preprocessing import LabelEncoder
from torch import tensor


class GNNQA(pl.LightningModule):
    def __init__(self, model=None):
        super().__init__()
        self.model = model


    def forward(self,
                input_ids,
                attention_mask,
                labels=None,
                edges=None
                ):

        print('Forward step')
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, edges=edges)
        print('bbbbbbbbbbbbbbbb')
        exit()
        return output.loss, output.logits

    def training_step(self, batch, batch_idx):

        pdb.set_trace()
        print('training step ')
        #'q_id = id domanda
        #title = question
        #selftext = text with additional information
        #document = vuoto
        #subreddit = direttive output es. explain like im five
        #answers
        #title_urls = url, vuoto
        #selftext_urls = url, vuoto
        #answers_urls = url delle risposte
        #answer_tok = tokenizzate answer

        input_ids = batch['input_ids']
        input_ids = tensor(input_ids, dtype=torch.int, device=self.device)
        attention_mask = batch['attention_mask']
        attention_mask = tensor(attention_mask, dtype=torch.int, device=self.device)
        labels = batch['answer_tok']['input_ids']
        edges = batch['graph']



        loss = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels, edges=edges)[0]

        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


class T5DataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, dataset, batch_size=1, args=None, name_mapping=None):
        super().__init__()

        self.tokenizer = tokenizer
        self.dataset = dataset
        self.batch_size = batch_size
        self.args = args

        dataset_columns = name_mapping.get(args.dataset, None)
        self.train_name = dataset_columns[0]
        self.eval_name = dataset_columns[1]
        self.test_name = dataset_columns[2]
        self.question_name = dataset_columns[3]
        self.answers_name = dataset_columns[4]

        self.dataset = self.dataset.map(lambda example: self.tokenizer(example[self.answers_name]['text'], padding='max_length', truncation=True, max_length=512, return_tensors='pt'))
        self.dataset = self.dataset.map(lambda example: {'answer_tok': self.tokenizer(example[self.question_name], padding='max_length', truncation=True, max_length=512, return_tensors='pt')})


    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size)

    """
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset['validation'], batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset['test'], batch_size=self.batch_size)"""


