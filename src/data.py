from datasets import load_dataset

loaded_datasets = {}


def get_dataset(dataset_name: str):
    global loaded_datasets

    if dataset_name in loaded_datasets:
        return loaded_datasets[dataset_name]
    else:
        if dataset_name == 'eli5':
            loaded_datasets[dataset_name] = load_dataset('eli5')
        elif dataset_name == 'trivia_qa':
            loaded_datasets[dataset_name] = load_dataset("trivia_qa")
        elif dataset_name == 'web_questions':
            loaded_datasets[dataset_name] = load_dataset("web_questions")
        elif dataset_name == 'natural_questions':
            loaded_datasets[dataset_name] = load_dataset("natural_questions")
        elif dataset_name == 'conceptnet':
            loaded_datasets[dataset_name] = load_dataset("peandrew/conceptnet_en_nomalized")

        return loaded_datasets[dataset_name]
