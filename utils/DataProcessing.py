import os
import requests
import tiktoken
import torch

class DataProcessing:
    def __init__(self, batch_size, block_size):
        self.block_size = block_size
        self.batch_size = batch_size
        self.datasets = {}

    def write_text_file(self, text_file):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        response = requests.get(url)

        # Save to file
        with open(text_file, 'w') as f:
            f.write(response.text)

    def generate_dataset(self, text_file, data_dir, split, tokenizer = tiktoken.get_encoding('gpt2')):    
        if not os.path.isfile(text_file):
            self.write_text_file(text_file=text_file)

        # create directory
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)
        
        with open(text_file, 'r') as f:
            data = f.read()

        tokenized_data = torch.Tensor(tokenizer.encode(data))
    
        n = int(len(tokenized_data)*split)
        self.datasets['train'] = tokenized_data[:n]
        self.datasets['valid'] = tokenized_data[n:]

        return self.datasets
        