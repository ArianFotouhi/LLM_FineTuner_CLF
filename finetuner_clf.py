from datasets import load_dataset
from transformers import AutoTokenizer, GPTNeoForSequenceClassification
from transformers import AutoTokenizer

# Load dataset
dataset = load_dataset("yelp_review_full")
print('dataset record example:', dataset["train"][1]) 


# Create tokenizer 
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

#
