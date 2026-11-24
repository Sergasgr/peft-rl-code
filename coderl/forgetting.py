# Copyright 2025-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified for this repository from peft's method_comparison/MetaMathQA (run.py and data.py, commit f471b33):
# the same measure of forgetting, packaged as two functions.
"""Forgetting: mean per-token loss on 100 Wikipedia articles, before and after training."""

import torch

FINEWIKI = "HuggingFaceFW/finewiki"


def load_texts(n: int = 100, revision: str | None = None) -> tuple[list[str], str]:
    from datasets import load_dataset
    from huggingface_hub import dataset_info

    sha = dataset_info(FINEWIKI, revision=revision).sha
    rows = load_dataset(FINEWIKI, split="train", streaming=True, revision=sha).take(n)  # no full shard download
    return [row["text"] for row in rows], sha


@torch.inference_mode()
def mean_token_loss(model, tokenizer, texts: list[str], batch_size: int = 4, max_length: int = 768) -> float:
    """Loss per token, so that long articles don't weigh more. Left padding; the mask selects the real tokens."""
    was_training = model.training
    model.eval()
    losses: list[float] = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(texts[start : start + batch_size], truncation=True, max_length=max_length)
        batch = tokenizer.pad(batch, return_tensors="pt", padding_side="left").to(model.device)
        logits = model(**batch).logits
        for logit, target, mask in zip(logits, batch["input_ids"], batch["attention_mask"], strict=True):
            num_tokens = int(mask.sum())
            logit, target = logit[-num_tokens:].float(), target[-num_tokens:]
            losses.extend(torch.nn.functional.cross_entropy(logit[:-1], target[1:], reduction="none").tolist())
    model.train(was_training)
    return float(torch.tensor(losses).mean())
