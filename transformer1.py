import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from datasets import load_dataset
from transformers import get_scheduler
import numpy as np
import time
import matplotlib.pyplot as plt

# Custom differential privacy optimizer (based on PyTorch optimizer)
class DPSGD(AdamW):
    def __init__(self, params, lr=1e-3, noise_multiplier=0.1, l2_norm_clip=1.0, **kwargs):
        super().__init__(params, lr=lr, **kwargs)
        self.noise_multiplier = noise_multiplier
        self.l2_norm_clip = l2_norm_clip

    def _compute_gradients(self, loss, model):
        # Manually calculate gradients and add noise
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                norm = torch.norm(p.grad)
                clip_factor = self.l2_norm_clip / (norm + 1e-6)
                p.grad = p.grad.clamp(max=clip_factor)  # Gradient Clipping
                noise = torch.normal(0, self.l2_norm_clip * self.noise_multiplier, size=p.grad.shape)
                p.grad += noise

# Setting up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading the TREC-6 dataset
dataset = load_dataset("trec", trust_remote_code=True)

# Load DistilBERT’s tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=6)
model.to(device)

# Data preprocessing
def preprocess_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)

encoded_dataset = dataset.map(preprocess_function, batched=True)
encoded_dataset = encoded_dataset.rename_column("coarse_label", "labels")
encoded_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Create DataLoader
train_dataloader = DataLoader(encoded_dataset["train"], shuffle=True, batch_size=16)
test_dataloader = DataLoader(encoded_dataset["test"], batch_size=16)

# Defining Noise Figure Ranges
noise_levels = [0.01, 0.1, 1.0]

results = {
    "noise_multiplier": [],
    "training_time": [],
    "accuracy": [],
    "response_time": [],
    "privacy_loss": []
}

# Training and evaluating the model
for noise_multiplier in noise_levels:
    optimizer = DPSGD(model.parameters(), lr=5e-5, noise_multiplier=noise_multiplier, l2_norm_clip=1.0)
    num_epochs = 3
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_epochs * len(train_dataloader),
    )

    # Record training time
    start_time = time.time()
    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            optimizer.zero_grad()
            optimizer._compute_gradients(loss, model)
            optimizer.step()
            lr_scheduler.step()
    training_time = time.time() - start_time

    # Record system response time
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
    response_time = time.time() - start_time

    # Calculate the accuracy on the test set
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == batch["labels"]).sum().item()
            total += len(batch["labels"])
    accuracy = correct / total

    # Calculating Privacy Loss
    privacy_loss = noise_multiplier 

    # results
    results["noise_multiplier"].append(noise_multiplier)
    results["training_time"].append(training_time)
    results["accuracy"].append(accuracy)
    results["response_time"].append(response_time)
    results["privacy_loss"].append(privacy_loss)

    print(f"Test accuracy: {accuracy:.4f}")

# plot graphs
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(results["noise_multiplier"], results["privacy_loss"], marker='o')
plt.title('Privacy Loss vs Noise Multiplier')
plt.xlabel('Noise Multiplier')
plt.ylabel('Privacy Loss')

plt.subplot(2, 2, 2)
plt.plot(results["noise_multiplier"], results["training_time"], marker='o')
plt.title('Training Time vs Noise Multiplier')
plt.xlabel('Noise Multiplier')
plt.ylabel('Training Time (seconds)')

plt.subplot(2, 2, 3)
plt.plot(results["noise_multiplier"], results["accuracy"], marker='o')
plt.title('Accuracy vs Noise Multiplier')
plt.xlabel('Noise Multiplier')
plt.ylabel('Accuracy')

plt.subplot(2, 2, 4)
plt.plot(results["noise_multiplier"], results["response_time"], marker='o')
plt.title('Response Time vs Noise Multiplier')
plt.xlabel('Noise Multiplier')
plt.ylabel('Response Time (seconds)')

plt.tight_layout()
plt.show()
