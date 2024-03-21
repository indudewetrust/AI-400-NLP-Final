import torch
from transformers import MarianMTModel, MarianTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

# Load the dataset
dataset = load_dataset("csv", data_files="lang/cleaned_ben.tsv", delimiter="\t")

# Load pre-trained model and tokenizer
base_model_name = "Helsinki-NLP/opus-mt-bn-en"
model = MarianMTModel.from_pretrained(base_model_name)
tokenizer = MarianTokenizer.from_pretrained(base_model_name)

# Define a function to preprocess the dataset
def preprocess_function(example):
    return tokenizer(example["English"], return_tensors="pt", padding=True, truncation=True)

# Preprocess the dataset
preprocessed_dataset = dataset.map(preprocess_function, batched=True)

# Fine-tune the model
def train(model, dataset, tokenizer):
    # Set up the training data
    train_data = dataset["train"]
    train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)

    # Set up the optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(5):
        total_loss = 0
        for batch in train_dataloader:
            inputs = batch["translation"]
            labels = batch["translation"]

            # Forward pass
            outputs = model(**inputs)
            logits = outputs.logits

            # Compute the loss
            loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} - Average Loss: {total_loss / len(train_dataloader)}")

# Call the training loop
train(model, preprocessed_dataset, tokenizer)

# Evaluate the fine-tuned model
def evaluate(model, dataset, tokenizer):
    # Set up the validation data
    val_data = dataset["validation"]
    val_dataloader = DataLoader(val_data, batch_size=16)

    # Set up the evaluation metrics
    total_loss = 0
    loss_fn = torch.nn.CrossEntropyLoss()  # Define the loss function

    # Evaluation loop
    for batch in val_dataloader:
        inputs = batch["translation"]
        labels = batch["translation"]

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits

        # Compute the loss
        loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))

        total_loss += loss.item()

    # Compute the average loss
    average_loss = total_loss / len(val_dataloader)
    print(f"Validation - Average Loss: {average_loss}")

# Call the evaluation function
evaluate(model, preprocessed_dataset, tokenizer)

# Save the fine-tuned model
model.save_pretrained("fine_tuned_ben_en_model")
tokenizer.save_pretrained("fine_tuned_ben_en_model")
