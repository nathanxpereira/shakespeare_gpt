# def tokenize_dataset(dataset, tokenizer, data_dir, context_len=128):
#     tokenized_dataset = {}
#     for key in dataset:
#         tokenized_dataset[key] = process_data(dataset[key], tokenizer, context_len=context_len)
#     tokenized_dataset = datasets.DatasetDict(tokenized_dataset)
#     tokenized_dataset.save_to_disk(data_dir)
#     return tokenized_dataset

# def extract_input_target(examples):
#     pass

# def process_data(dataset, tokenizer, context_len=128):
#     tokenized_dataset = dataset.map(lambda examples: tokenizer(examples['text'], truncation=True, padding='max_length', max_length=context_len+1), remove_columns=['text'])
#     tokenized_dataset = tokenized_dataset.add_column("labels", tokenized_dataset["input_ids"].copy())
#     tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
#     return tokenized_dataset