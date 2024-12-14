from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import argparse
import os

file_dir = "Sample Projects\\Shakespeare GPT"
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using cuda")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(os.path.join(file_dir,'shakespeare_gpt2'))
model = GPT2LMHeadModel.from_pretrained(os.path.join(file_dir,'shakespeare_gpt2'))

# Define the pad_token as eos_token
tokenizer.pad_token = tokenizer.eos_token

# Update the model's configuration to recognize the pad_token_id
model.config.pad_token_id = tokenizer.pad_token_id

model.to(device)
model.eval()


def generate_text(prompt, device, max_length=100):
    # Encode the prompt and generate attention_mask
    encoded = tokenizer(
        prompt,
        return_tensors='pt',
        padding=False,  # No padding needed for single inputs
        truncation=True,
        max_length=512  # Adjust based on model's max input length
    )
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,  # Pass the attention_mask
            max_length=max_length,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id  # Ensure pad_token_id is set
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser(description="Generate text based on a prompt.")
    parser.add_argument("--prompt", nargs="?", default=None, help="The prompt to generate text from.")
    args = parser.parse_args()

    if args.prompt:
        print(f"Prompt: {args.prompt}")
        print(generate_text(args.prompt)[len(args.prompt):], sep="")
    else:
        prompts = [
            "O Romeo, Romeo! wherefore art",
            "All the world's a",
            "Is this a dagger which I"
        ]

        for prompt in prompts:
            print(f"Prompt: {prompt}")
            print(generate_text(prompt, device='cuda')[len(prompt):], sep="")
            print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    main()