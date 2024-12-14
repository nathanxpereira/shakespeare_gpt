import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from datasets import load_dataset


import torch
import torch.nn as nn

# Step 1: Load Dataset
if __name__=="__main__":
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")  # Replace with your dataset
    train_data = dataset["train"]["text"]
    val_data = dataset["validation"]["text"]

    # Step 2: Tokenizer and Preprocessing
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def preprocess(data):
        return tokenizer(data, truncation=True, padding=True, max_length=512, return_tensors="pt")

    train_encodings = [preprocess(text) for text in train_data]
    val_encodings = [preprocess(text) for text in val_data]

    # Step 3: Define Dataloader
    train_loader = DataLoader(train_encodings, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_encodings, batch_size=8)

    # Step 4: Initialize Model
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))  # Adjust tokenizer size if new tokens are added
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Step 5: Define Optimizer and Loss
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Step 6: Training Loop
    epochs = 3
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            inputs = batch["input_ids"].squeeze().to(device)
            attention_mask = batch["attention_mask"].squeeze().to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs, attention_mask=attention_mask, labels=inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1} completed. Loss: {loss.item()}")

    # Step 7: Save Model
    model.save_pretrained("./trained_gpt2_model")
    tokenizer.save_pretrained("./trained_gpt2_model")



    # Example data
    batch_size = 2
    seq_length = 4
    vocab_size = 10

    # Model outputs: logits (raw scores, not probabilities)
    logits = torch.randn(batch_size, seq_length, vocab_size)  # Random scores
    targets = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])  # True token indices

    # Reshape logits and targets for CrossEntropyLoss
    logits = logits.view(-1, vocab_size)  # Shape: (batch_size * seq_length, vocab_size)
    targets = targets.view(-1)           # Shape: (batch_size * seq_length)

    # Loss function
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, targets)

    print(f"Loss: {loss.item()}")
