import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer


TOTAL_TOKENS = 100_000_000
MAX_PROMPT = 768
MAX_ANSWER = 256


class ChatDataset(IterableDataset):
    def __init__(self, tokenizer, max_prompt=MAX_PROMPT, max_answer=MAX_ANSWER, max_tokens=TOTAL_TOKENS):
        self.tokenizer = tokenizer
        self.max_prompt = max_prompt
        self.max_answer = max_answer
        self.max_tokens = max_tokens
        self.dataset = load_dataset(
            "Open-Orca/OpenOrca",
            split="train",
            streaming=True,
        )

    def __iter__(self):
        total_tokens = 0
        for example in self.dataset:
            question = example.get("question", "")
            response = example.get("response", "")
            system = example.get("system_prompt", "")

            if not question or not response:
                continue

            if system:
                prompt_text = f"{system}\n\n{question}"
            else:
                prompt_text = question

            prompt_ids = self.tokenizer(
                prompt_text,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_prompt,
            )["input_ids"]

            answer_ids = self.tokenizer(
                response,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_answer,
            )["input_ids"]

            if len(prompt_ids) < 2 or len(answer_ids) < 1:
                continue

            total_tokens += len(prompt_ids) + len(answer_ids)
            yield {
                "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
                "answer_ids": torch.tensor(answer_ids, dtype=torch.long),
            }

            if total_tokens >= self.max_tokens:
                break


def collate_fn(batch):
    B = len(batch)
    max_p = max(x["prompt_ids"].shape[0] for x in batch)
    max_a = max(x["answer_ids"].shape[0] for x in batch)

    prompt_ids = torch.full((B, max_p), 0, dtype=torch.long)
    answer_ids = torch.full((B, max_a), 0, dtype=torch.long)

    for i, ex in enumerate(batch):
        pl = ex["prompt_ids"].shape[0]
        al = ex["answer_ids"].shape[0]
        prompt_ids[i, :pl] = ex["prompt_ids"]
        answer_ids[i, :al] = ex["answer_ids"]

    return {
        "prompt_ids": prompt_ids,
        "answer_ids": answer_ids,
    }


def create_dataloader(tokenizer, batch_size=4, max_tokens=TOTAL_TOKENS):
    dataset = ChatDataset(tokenizer, max_tokens=max_tokens)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)


def create_eval_dataloader(tokenizer, batch_size=4, max_tokens=5_000_000):
    dataset = ChatDataset(tokenizer, max_tokens=max_tokens)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
