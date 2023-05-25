from datasets import load_dataset


def get_dataset(dataset_name: str):
    if dataset_name == 'eli5':
        return load_dataset('eli5')
    elif dataset_name == 'trivia_qa':
        return load_dataset("trivia_qa")
    elif dataset_name == 'web_questions':
        return load_dataset("web_questions")
    elif dataset_name == 'natural_questions':
        return load_dataset("natural_questions")
    elif dataset_name == 'conceptnet':
        return load_dataset("peandrew/conceptnet_en_nomalized")
