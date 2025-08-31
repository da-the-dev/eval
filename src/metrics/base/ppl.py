import torch

from datasets import Dataset, DatasetDict
from tqdm import tqdm
from transformers import Any


def ppl(
    model: Any,
    tokenizer: Any,
    dataset: Dataset | DatasetDict,
    max_length=1024,
    stride=512,
):
    mode = model.training
    model_device = model.device
    model.training = mode

    model.eval()

    tokenized_dataset = tokenizer(
        ''.join([d['text'] for d in dataset]),
        return_tensors='pt',
        max_length=max_length,
        stride=stride,
        truncation=True,
        padding=True,
        return_overflowing_tokens=True,
    )

    total_loss = 0
    total_tokens = 0
    for i in tqdm(range(tokenized_dataset['input_ids'].shape[0])):
        input_ids_chunk = (
            tokenized_dataset['input_ids'][i].unsqueeze(0).to(model_device)
        )
        attention_mask_chunk = (
            tokenized_dataset['attention_mask'][i].unsqueeze(0).to(model_device)
        )

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids_chunk,
                attention_mask=attention_mask_chunk,
                labels=input_ids_chunk,
            )
        chunk_average_loss = outputs.loss

        attention_mask_chunk = tokenized_dataset['attention_mask'][i].to(model_device)

        num_real_tokens_in_chunk = attention_mask_chunk.sum()

        chunk_total_loss = chunk_average_loss * num_real_tokens_in_chunk

        total_loss += chunk_total_loss
        total_tokens += num_real_tokens_in_chunk

    final_average_loss = total_loss / total_tokens

    return torch.exp(final_average_loss)
