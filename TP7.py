from datasets import load_dataset, concatenate_datasets, DatasetDict, ClassLabel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

ds1 = load_dataset("zeroshot/twitter-financial-news-sentiment")
ds2 = load_dataset("nickmuchi/financial-classification")


ds2 = DatasetDict({
    split: ds.rename_column("labels", "label")
    for split, ds in ds2.items()
})
train_ds = concatenate_datasets([ds1["train"], ds2["train"]])
test_ds = concatenate_datasets([ds1["validation"], ds2["test"]])

D={
        "train": train_ds,
        "test": test_ds
    }

def train_model(model_name, dataset, batch_size=16, num_epochs=3):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    def preprocess(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length")

    tokenized_train = dataset["train"].map(preprocess, batched=True)
    tokenized_test = dataset["test"].map(preprocess, batched=True)

    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    training_args = TrainingArguments(
        output_dir=f"./{model_name.replace('/', '_')}_results",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        report_to="none",
        disable_tqdm=False,
        logging_first_step=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        #compute_metrics=compute_metrics
    )

    trainer.train()
    results = trainer.evaluate()
    print(f"Résultats pour {model_name}:", results)

    model.save_pretrained(f"./{model_name.replace('/', '_')}_finetuned")
    tokenizer.save_pretrained(f"./{model_name.replace('/', '_')}_finetuned")

#train_model("bert-base-uncased", D)

train_model("ProsusAI/finbert", D)