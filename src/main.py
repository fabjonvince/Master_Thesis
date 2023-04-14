import argparse

from transformers import T5Tokenizer

from preprocess import text_to_graph, print_triplets
from data import get_dataset
from model import GNNQA, T5DataModule
from t5 import T5KILForConditionalGeneration
from pytorch_lightning import Trainer

argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str, default='eli5', help='Dataset to use')
argparser.add_argument('--train_samples', type=int, default=10, help='Number of train samples')
argparser.add_argument('--layer_kil', type=int, default=[1, 2], help='Layers with KIL')
name_mapping = {
"eli5": ("train_eli5", "validation_eli5", "test_eli5", "title", "answers")
}


def main(args):
    #pdb.set_trace()
    print("In Main")

    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    new_tokens = ['<S-ENT>', '<T-ENT>']
    tokenizer.add_tokens(new_tokens)


    #load dataset
    dataset = get_dataset(args.dataset)
    dataset_columns = name_mapping.get(args.dataset, None)
    train_name = dataset_columns[0]
    eval_name = dataset_columns[1]
    test_name = dataset_columns[2]
    question_name = dataset_columns[3]
    answers_name = dataset_columns[4]

    # prova con una singola frase
    #triplets = text_to_graph(2, dataset[train_name][0][question_name])

    # dataset sampling
    dataset[train_name] = dataset[train_name].shuffle(seed=42).select(range(args.train_samples))
    dataset[eval_name] = dataset[eval_name].shuffle(seed=42).select(range(10))

    # dataset preprocessing
    # odeificare text_to_graph per avere in input una sola frase
    # aggiungere al grafo creato un super nodo che rappresenta la domanda
    dataset[train_name] = dataset[train_name].map(lambda example: {'graph': text_to_graph(2, example[question_name])})
    dataset[eval_name] = dataset[eval_name].map(lambda example: {'graph': text_to_graph(2, example[question_name])})

    # tokenization
    #dataset[train_name] = dataset[train_name].map(lambda example: tokenizer(example[question_name], padding='max_length', truncation=True, max_length=512), batched=True)
    #dataset[eval_name] = dataset[eval_name].map(lambda example: tokenizer(example[question_name], padding='max_length', truncation=True, max_length=512), batched=True)

    #dataload
    train_dataload = T5DataModule(tokenizer, dataset[train_name], batch_size=1, args=args, name_mapping=name_mapping)

    #model creation
    model = T5KILForConditionalGeneration.from_pretrained('t5-base')
    gnnqa = GNNQA(model)
    trainer_args = {'max_epochs': 1, 'gpus': 1}

    trainer = Trainer()
    trainer.fit(model=gnnqa, train_dataloaders=train_dataload)#, val_dataloaders=dataset[eval_name])

    trainer.train()

if __name__ == '__main__':
    args = argparser.parse_args()
    main(args)


