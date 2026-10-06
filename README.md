# Qwen3-4B Math Reasoning

Improving the math reasoning of **Qwen3-4B-Thinking-2507** through prompt engineering and parameter-efficient fine-tuning (SFT + GRPO with LoRA), evaluated on 1,126 problems (375 multiple choice, 751 free-response) ranging from high school to graduate level.

## Results

| Notebook | MCQ | Free-response | Overall |
|---|---|---|---|
| `01_baseline.ipynb` | 35.20% | 49.53% | 44.76% |
| `02_prompt_engineering.ipynb` | 39.20% | 51.26% | **47.25%** |

## Approach

**Prompt engineering**
* Diagnosed that ~80% of baseline MCQ responses (and ~38% of free-response) hit the token limit before writing a final `\boxed{}` answer.
* Built separate prompts for MCQ, free-response, and questions with embedded A/B/C choices; all require brief reasoning and a final `\boxed{}`, and free-response prompts count `[ANS]` blanks.

**Fine-tuning (SFT + GRPO)**
* **Model:** `unsloth/Qwen3-4B-Base` with LoRA adapters (rank 32) via Unsloth.
* **SFT:** trained on curated OpenMathReasoning examples to teach a structured reason-then-answer format.
* **GRPO:** reinforcement learning on NuminaMath-CoT with reward functions that score answer correctness, numeric equivalence, output format, and multi-letter MCQ answers with partial credit. Responses without a clear final answer are penalized, pushing the model toward committed, checkable answers.

## Hardware & Performance

* **GPU Used:** NVIDIA A100 (Google Colab). A T4 (15GB) runs out of memory.
* **Approximate Inference Time:** several hours per full evaluation run (Hugging Face `generate`, ~4k max tokens).

## Setup Instructions

Place the dataset at `data/public.jsonl` (one JSON object per line with `id`, `question`, `answer`, and `options` for multiple choice) and keep `judger.py` in the same folder as the notebook.

The first cell of each notebook installs dependencies and restarts the kernel. The crash is expected; run the rest of the cells in order afterward.

## How to Reproduce Results

Open either notebook in Colab on an A100 runtime and run all cells. `02_prompt_engineering.ipynb` saves after every batch and resumes automatically if the session disconnects. Its final cell re-scores everything from the saved results file.

## Repository Structure

```
judger.py                      answer checker (letter match + sympy equivalence)
notebooks/
  01_baseline.ipynb            starter prompts
  02_prompt_engineering.ipynb  improved prompts + resume logic
```

`judger.py` is a rewrite of the checker used for the original runs, so re-scoring may differ by about a point.
