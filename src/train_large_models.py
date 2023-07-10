from functools import partial

import datasets
import transformers
import argparse

from datasets import load_from_disk
from transformers import TrainingArguments, Seq2SeqTrainingArguments

from data import name_mapping
from tools import get_rouge_scores_for_hftrainer
from peft import prepare_model_for_int8_training

supported_models={
    'google/t5-3b-ssm-nq': transformers.AutoModelForSeq2SeqLM,
    'google/t5-11b-ssm-nq': transformers.AutoModelForSeq2SeqLM,
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_dataset_from', type=str, default=None, required=True)
    parser.add_argument('--dataset', type=str, default='eli5', required=True)
    parser.add_argument('--model', type=str, default='t5-3b', required=True)
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--train_samples', type=int, default=None)
    parser.add_argument('--val_samples', type=int, default=None)
    parser.add_argument('--test_samples', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--save_model_to', type=str, default='pretrained_large_models')

def tokenize_dataset(dataset, tokenizer, question_name, answers_name, is_t5=False):
    def tokenize_function(examples):
        batch = tokenizer(examples[question_name])
        labels = tokenizer(examples[answers_name])['input_ids']
        batch['labels'] = labels

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=[question_name, answers_name]
    )
    tokenized_datasets = tokenized_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=[question_name, answers_name]
    )
    tokenized_datasets = tokenized_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=[question_name, answers_name]
    )
    return tokenized_datasets


def main(args):
    print("In Main")
    print("Loading Dataset")
    dataset = load_from_disk(args.load_dataset_from)
    dataset_columns = name_mapping.get(args.dataset, None)
    train_name = dataset_columns[0]
    val_name = dataset_columns[1]
    test_name = dataset_columns[2]
    question_name = dataset_columns[3]
    answers_name = dataset_columns[4]
    if args.model is ['t5-3b', 't5-11b']:
        dataset[train_name] = dataset[train_name].map(
            lambda example: {'question': 'question: ' + example['question']})
        dataset[val_name] = dataset[val_name].map(lambda example: {'question': 'question: ' + example['question']})
        dataset[test_name] = dataset[test_name].map(lambda example: {'question': 'question: ' + example['question']})

    print("Dataset Loaded")

    # now I check if args.train_samples val_samples and test_samples are not None and I sample the dataset
    if args.train_samples is not None:
        dataset[train_name] = dataset[train_name].select(range(args.train_samples))
    if args.val_samples is not None:
        dataset[val_name] = dataset[val_name].select(range(args.val_samples))
    if args.test_samples is not None:
        dataset[test_name] = dataset[test_name].select(range(args.test_samples))

    print("Loading Model")
    assert args.model in supported_models, f"Model {args.model} not supported"
    if args.use_lora:
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(args.model, load_in_8bit=True, device_map='auto')
        model = prepare_model_for_int8_training(model)
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(args.model)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)

    print("Model Loaded")

    print("Tokenizing Dataset")
    tokenized_datasets = tokenize_dataset(dataset, tokenizer, question_name, answers_name)
    print("Dataset Tokenized")

    print("Training Model")
    # I want the trainer log on wandb
    train_args = Seq2SeqTrainingArguments(
        f"{args.model}-finetuned-{args.dataset}",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=3,
        predict_with_generate=True,
        weight_decay=0.01,
        push_to_hub=False,
        save_total_limit=1,
        report_to="wandb",
        gradient_accumulation_steps=args.batch_size,
        gradient_checkpointing=True,
    )
    trainer = transformers.Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_datasets[train_name],
        eval_dataset=tokenized_datasets[val_name],
        tokenizer=tokenizer,
        compute_metrics=partial(get_rouge_scores_for_hftrainer, tokenizer=tokenizer)
    )

    trainer.train()
    print("Model Trained")
    trainer.save(args.save_model_to + f"/{args.model}-finetuned-{args.dataset}")
    print("Model Saved")

    print("Evaluating Model")
    eval_results = trainer.evaluate(tokenized_datasets[test_name])
    print("Model Evaluated")




if __name__ == '__main__':
    args = get_args()
    main(args)