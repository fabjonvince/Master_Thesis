import argparse
import pdb

from transformers import T5Tokenizer

from preprocess import text_to_graph, print_triplets
from src.data import get_dataset
from src.model import GNNQA
from src.t5 import T5KILForConditionalGeneration
from pytorch_lightning import Trainer

argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str, default='eli5', help='Dataset to use')
argparser.add_argument('--train_samples', type=int, default=1000, help='Number of train samples')


def main(args):
    pdb.set_trace()
    print("In Main")
    sent1 = "Where is born Barack Obama?"
    sent2 = "Obama was born in Honolulu, Hawaii. After graduating from Columbia University in 1983, he worked as a community organizer in Chicago. In 1988, he enrolled in Harvard Law School, where he was the first black president of the Harvard Law Review."
    #graph = text_to_graph2(3, sent1, sent2)
    graph = text_to_graph(3, sent1)
    print_triplets(graph)

    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    #load dataset
    dataset = get_dataset(args.dataset)

    # dataset sampling
    dataset['train'] = dataset['train'].shuffle(seed=42).select(range(args.train_samples))
    dataset['validation'] = dataset['validation'].shuffle(seed=42).select(range(100))

    # dataset preprocessing
    # odeificare text_to_graph per avere in input una sola frase
    # aggiungere al grafo creato un super nodo che rappresenta la domanda
    dataset['train'] = dataset['train'].map(lambda example: {'graph': text_to_graph(3, example['question'])})
    dataset['validation'] = dataset['validation'].map(lambda example: {'graph': text_to_graph(3, example['question'])})

    # tokenization
    dataset['train'] = dataset['train'].map(lambda example: tokenizer(example['question'], padding='max_length', truncation=True, max_length=512), batched=True)
    dataset['validation'] = dataset['validation'].map(lambda example: tokenizer(example['question'], padding='max_length', truncation=True, max_length=512), batched=True)


    #model creation
    model = T5KILForConditionalGeneration.from_pretrained('google/t5-v1_1-base')
    gnnqa = GNNQA(model)
    trainer_args = {'max_epochs': 1, 'gpus': 1}
    trainer = Trainer(model=gnnqa, args=trainer_args, train_dataset=dataset['train'], eval_dataset=dataset['validation'])

    trainer.train()

if __name__ == '__main__':
    args = argparser.parse_args()
    main(args)