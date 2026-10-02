# Making Qwen3-4B Better at Math

This is a project where I tried to get **Qwen3-4B-Thinking-2507**, a pretty small 4B-parameter model, to answer more math questions correctly. The questions range from high school to grad level, and they're a mix of multiple choice and free-response (some with several `[ANS]` blanks that all have to be right for the question to count).

## Results

Everything below was scored on all 1,126 questions in the evaluation set (375 multiple choice, 751 free-response).

| Notebook | MCQ | Free-response | Overall |
|---|---|---|---|
| `01_baseline.ipynb` (simple starter prompts) | 35.20% | 49.53% | **44.76%** |
| `02_prompt_engineering.ipynb` (my prompts) | 39.20% | 51.26% | **47.25%** |

So prompt engineering alone got about **+2.5 points**, with most of the gain on multiple choice.

## What I found

The biggest thing I learned came from actually reading the model's outputs instead of just looking at the score. In the baseline, around **80% of the multiple choice responses were getting cut off** before the model ever wrote its final `\boxed{}` answer. The model would overthink, run out of tokens, and get marked wrong even when it was on the right track. About 38% of the free-response answers had the same problem.

So my prompts were built to fight that:

- Three different system prompts depending on the question type: real multiple choice, free-response, and free-response questions that secretly have A/B/C choices written into the text.
- Every prompt tells the model to reason briefly and **always** finish with `\boxed{}`, even if it isn't sure.
- The free-response prompt tells it to count the `[ANS]` blanks and give exactly that many answers.
- I lowered the max generation length from 8192 to ~4096 tokens, which also made full runs actually finishable on the hardware I had.

## What I tried that didn't work

I also built an SFT + GRPO (reinforcement learning) fine-tuning pipeline with LoRA. It actually made things worse (about 35% overall). When I looked into it, the cause was a format mismatch: the training taught the model one answer format while the evaluation expected `\boxed{}`, so a lot of answers leaked training tags or rambled past the token limit. Lesson learned: keep the training format, eval prompt, and grader all speaking the same format, and check an early checkpoint against the baseline before committing hours to a full run.

## How to run it

The notebooks were written for **Google Colab with an A100**. A T4 (15GB) runs out of memory on this model.

1. Put the question file at `data/public.jsonl`. Each line is a JSON object with `id`, `question`, `answer`, and `options` (only for multiple choice).
2. Make sure `judger.py` from this repo is in the same folder you run the notebook from (in Colab, upload it alongside the data).
3. Switch the Colab runtime to an A100 and run the first cell. It installs packages and then intentionally restarts the kernel, so that crash is expected.
4. Run the rest of the cells in order.

`02_prompt_engineering.ipynb` saves results after every batch and backs them up to Google Drive, so if Colab disconnects you can rerun it and it picks up where it left off. The last cell re-scores everything from the saved file so the numbers aren't affected by any grading hiccups during the run.

`judger.py` checks multiple choice by letter and free-response answers by symbolic/numeric equivalence with sympy (so `5/8`, `0.625`, and `\frac{5}{8}` all count as the same). It's a rewrite of the checker I originally scored with, so re-running might land a point or so off the table above.

## Repo layout

```
judger.py                      answer checker (letter match + sympy equivalence)
notebooks/
  01_baseline.ipynb            starter prompts, full eval-set run
  02_prompt_engineering.ipynb  my prompts + resume/backup logic
```

## Tools

Python, PyTorch, Hugging Face Transformers, sympy, and Google Colab.
