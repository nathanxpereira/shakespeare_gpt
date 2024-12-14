import requests
import datasets

def generate_dataset(text_file, data_dir, url):
    response = requests.get(url)
    raw_dataset = response.text

    data = response.text

    # Save to file
    with open(text_file, 'w') as f:
        f.write(data)
    
    # tokenize data
    raw_dataset = datasets.Dataset.from_dict({'text': data.split('\n')})

    filtered_dataset = raw_dataset.filter(lambda example: example["text"] and len(example["text"].strip()) > 0)

    train_validtest = filtered_dataset.train_test_split(0.1)
    valid_test = train_validtest['test'].train_test_split(0.5)

    dataset = datasets.DatasetDict({'train': train_validtest['train'],
                                    'valid': valid_test['train'],
                                    'test': valid_test['test']})
    # filter dataset from empty entries

    dataset.save_to_disk(data_dir)

def process_data(dataset, tokenizer):
    tokenized_dataset = dataset.map(lambda examples: tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128), remove_columns=['text'])
    tokenized_dataset = tokenized_dataset.add_column("labels", tokenized_dataset["input_ids"].copy())
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return tokenized_dataset

if __name__ == "__main__":
    file_dir = r"Sample Projects\Shakespeare GPT"
    generate_dataset(file_dir=file_dir)