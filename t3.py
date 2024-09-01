import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from datasets import load_dataset
from transformers import get_scheduler

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载TREC-6数据集，添加 trust_remote_code=True 参数
dataset = load_dataset("trec", trust_remote_code=True)

# 加载DistilBERT的tokenizer和模型
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=6)
model.to(device)

# 数据预处理
def preprocess_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)

encoded_dataset = dataset.map(preprocess_function, batched=True)

# 将 'coarse_label' 列重命名为 'labels'
encoded_dataset = encoded_dataset.rename_column("coarse_label", "labels")
encoded_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# 创建DataLoader
train_dataloader = DataLoader(encoded_dataset["train"], shuffle=True, batch_size=64)
test_dataloader = DataLoader(encoded_dataset["test"], batch_size=64)

# 优化器和学习率调度
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# 训练函数
def train(model, dataloader, optimizer, scheduler):
    model.train()
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        print(f"Training loss: {loss.item()}")

# 评估函数
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == batch["labels"]).sum().item()
            total += len(batch["labels"])

    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy:.4f}")
    return accuracy

# 执行训练和评估
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}")
    train(model, train_dataloader, optimizer, lr_scheduler)
    evaluate(model, test_dataloader)

# 保存模型
model.save_pretrained("./saved_model_distilbert_trec")
tokenizer.save_pretrained("./saved_model_distilbert_trec")
