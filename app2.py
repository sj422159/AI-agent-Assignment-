import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

# Load dataset
df = pd.read_parquet('./train-00000-of-00001.parquet')

# Split dataset into two parts
train_df, finetune_df = train_test_split(df, test_size=0.5, random_state=42)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModelForSequenceClassification.from_pretrained("answerdotai/ModernBERT-base")

# Tokenize datasets
train_encodings = tokenizer(list(train_df['text']), truncation=True, padding=True)
finetune_encodings = tokenizer(list(finetune_df['text']), truncation=True, padding=True)

# Create torch datasets
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = Dataset(train_encodings, list(train_df['label']))
finetune_dataset = Dataset(finetune_encodings, list(finetune_df['label']))

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=finetune_dataset
)

# Train the model
trainer.train()

# Evaluate the pretrained model
pretrained_results = trainer.evaluate()

# Fine-tune the model
trainer.train()

# Evaluate the fine-tuned model
finetuned_results = trainer.evaluate()

# Load and fine-tune ModernBERT-large model
large_model = AutoModelForSequenceClassification.from_pretrained("answerdotai/ModernBERT-large")
large_trainer = Trainer(
    model=large_model,
    args=training_args,
    train_dataset=Dataset(tokenizer(list(df['text']), truncation=True, padding=True), list(df['label'])),
    eval_dataset=finetune_dataset
)

# Fine-tune the large model
large_trainer.train()

# Evaluate the fine-tuned large model
large_results = large_trainer.evaluate()

# Generate classification reports and ROC curves
def generate_metrics(trainer, dataset):
    predictions, labels, _ = trainer.predict(dataset)
    preds = predictions.argmax(-1)
    report = classification_report(labels, preds)
    fpr, tpr, _ = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)
    return report, fpr, tpr, roc_auc

pretrained_report, pretrained_fpr, pretrained_tpr, pretrained_auc = generate_metrics(trainer, finetune_dataset)
finetuned_report, finetuned_fpr, finetuned_tpr, finetuned_auc = generate_metrics(trainer, finetune_dataset)
large_report, large_fpr, large_tpr, large_auc = generate_metrics(large_trainer, finetune_dataset)

# Print results
print("Pretrained Model Classification Report:\n", pretrained_report)
print("Finetuned Model Classification Report:\n", finetuned_report)
print("Large Model Classification Report:\n", large_report)

# Plot ROC curves
import matplotlib.pyplot as plt

plt.figure()
plt.plot(pretrained_fpr, pretrained_tpr, color='darkorange', lw=2, label='Pretrained ROC curve (area = %0.2f)' % pretrained_auc)
plt.plot(finetuned_fpr, finetuned_tpr, color='blue', lw=2, label='Finetuned ROC curve (area = %0.2f)' % finetuned_auc)
plt.plot(large_fpr, large_tpr, color='green', lw=2, label='Large Model ROC curve (area = %0.2f)' % large_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()