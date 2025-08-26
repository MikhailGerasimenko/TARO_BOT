import json
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# 1. Загрузка датасета
with open('deepseek_json_20250707_b9011f.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

dataset = Dataset.from_list(data)
split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split['train']
test_dataset = split['test']

# 2. Токенизация
model_name = 'Priyanka-Balivada/Russian-BERT'
tokenizer = BertTokenizer.from_pretrained(model_name)

def preprocess(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=64)

train_dataset = train_dataset.map(preprocess, batched=True)
test_dataset = test_dataset.map(preprocess, batched=True)

# 3. Модель
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2, ignore_mismatched_sizes=True)

# 4. Метрики
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds)
    }

# 5. Аргументы обучения
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir='./logs',
)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# 7. Обучение
trainer.train()

# 8. Сохранение модели
trainer.save_model('./russian-bert-tarot-classifier')
tokenizer.save_pretrained('./russian-bert-tarot-classifier') 