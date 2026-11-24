# peft-rl-code

Which parameter-efficient fine-tuning (PEFT) methods work for reinforcement learning on code, when the reward checks both correctness (unit tests) and quality (lint and complexity)? Every experiment runs on a single 16 GB GPU.

## Why

This started as [CodeAlign](https://github.com/Sergasgr/CodeAlign/tree/815eb9cd7e7c105476671bc4f0768cf199f387d7), an SFT + DPO pipeline that ranked preference pairs by execution and static analysis. Over-trained, DPO learned to game "runs without errors": 34 of 164 HumanEval solutions did nothing, most of them the whole function commented out. RL work on code quality sees the same gaming when quality is rewarded on its own ([REAL](https://arxiv.org/abs/2505.22704), [Kadam, 2026](https://arxiv.org/abs/2610.38977)), and doesn't compare PEFT methods. Those have been compared under RL only on math ([Yin et al., ICML 2026](https://proceedings.mlr.press/v306/yin26h.html)).

## Questions

1. Do the math results (DoRA above LoRA, PiSSA collapsing, VeRA too small to learn) hold on code?
2. Does the adapter make a quality reward easier or harder to game?
3. Does quality measured outside the reward improve too, or only the numbers the reward sees?

## Setup

| | |
|---|---|
| Model | Qwen2.5-Coder-1.5B-Instruct, bf16 (pilot: 0.5B) |
| RL | GRPO from TRL: 300 steps × 8 prompts × 8 samples, no KL term |
| Data | [KodCode-Light-RL-10K](https://huggingface.co/datasets/KodCode/KodCode-Light-RL-10K): function tasks with pytest tests, reference solutions re-run, fixed train / valid (200) / test (500) |
| Rewards | `tests` · `tests+quality` (quality counts only when the tests pass) · `quality` (a probe: nothing checks correctness) |
| Methods | LoRA, DoRA, PiSSA, OFT, VeRA, IA³, TinyLoRA, full fine-tuning |
| Learning rate | a 3-point grid per method, chosen on the valid split |
| Evaluation | KodCode test, HumanEval+ and MBPP+ (EvalPlus, pinned); ruff and cyclomatic complexity (in the reward); pylint and maintainability index (outside it); no-op rate; forgetting |
| Baseline | the untrained model, greedy and best-of-8 reranked by the quality score |

## Run

```bash
uv sync              # environment from uv.lock
make test            # fast tests, no GPU
make data            # data/kodcode/ (once)
make baseline        # untrained models
make experiments     # experiments/ and queue.txt from methods.json
make night           # runs queue.txt overnight; results/*.json
make table           # RESULTS.md
```

`make run EXP=experiments/lora/lr1e-5-tests` runs a single experiment, and `compare.py` tells whether two configurations differ on the same tasks (exact McNemar, or a paired bootstrap over seeds). Each experiment is a folder with `adapter_config.json` (none for full fine-tuning) and `training_params.json`, the settings that differ from `default_training_params.json`, as in [peft's method comparison](https://github.com/huggingface/peft/tree/main/method_comparison).

## Results

None yet. They will be in [RESULTS.md](RESULTS.md).

## License

MIT. `coderl/forgetting.py` is adapted from peft and `coderl/evalplus.py` copies EvalPlus's chat prompt, both Apache-2.0.
