from datasets import load_from_disk
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer

from model import GNNQA
from main import argparser, name_mapping
from preprocess import create_memory
from t5 import T5GNNForConditionalGeneration


args = argparser.parse_args()

def main(args):
    print("In Main")
    dataset_columns = name_mapping.get(args.dataset, None)
    train_name = dataset_columns[0]
    eval_name = dataset_columns[1]
    test_name = dataset_columns[2]
    question_name = dataset_columns[3]
    answers_name = dataset_columns[4]

    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    dataset = load_from_disk('dataset/eli5_100_conceptnet')

    rels = [key for d in (dataset[train_name]['relations'], dataset[eval_name]['relations'], dataset[test_name]['relations']) for key in d.keys()]

    nodes = [key for d in (dataset[train_name]['nodes'], dataset[eval_name]['nodes'], dataset[test_name]['nodes']) for key in d.keys()]

    # Load a pretrained model with all-MiniLM-L12-v2 checkpoint
    model = SentenceTransformer('all-MiniLM-L12-v2')
    memory_nodes = create_memory(model, nodes, {'convert_to_tensor': True})
    memory_rels = create_memory(model, rels, {'convert_to_tensor': True})

    setattr(args, 'nrel', len(memory_rels))
    setattr(args, 'nnodes', len(memory_nodes))
    setattr(args, 'gnn_embs_size', args.sentence_transformer_embedding_size)

    model = T5GNNForConditionalGeneration.from_pretrained('t5-base', args)
    gnnqa = GNNQA.load_from_checkpoint('checkpoints/gnnqa-epoch=00-val_loss=0.00.ckpt', model=model, memory_rels=memory_rels, memory_nodes=memory_nodes)
    trainer_args = {'max_epochs': args.max_epochs, 'gpus': args.gpus}

    early_stopper = EarlyStopping(monitor='val_loss', patience=args.patience, mode='min')
    md_checkpoint = ModelCheckpoint(monitor='val_loss', save_top_k=args.save_top_k, mode='min', dirpath='checkpoints', filename='gnnqa-{epoch:02d}-{val_loss:.2f}')
    trainer = Trainer(trainer_args, callbacks=[early_stopper, md_checkpoint])
    #trainer.fit(model=gnnqa, ckpt_path='checkpoints/gnnqa-epoch=00-val_loss=0.00.ckpt')
    # trainer.load_checkpoint(gnnqa, 'checkpoints/gnnqa-epoch=00-val_loss=0.00.ckpt')

    results = trainer.predict(model=gnnqa, test_dataloaders=dataset[test_name])

    print(results)


if __name__ == '__test__':
    args = argparser.parse_args()
    main(args)

