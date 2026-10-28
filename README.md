# Qwen3-4B Math Reasoning

Improving the math accuracy of **Qwen3-4B-Thinking-2507** using only prompt engineering (PE), supervised fine-tuning (SFT) and reinforcement learning (GRPO), with no changes to the model's sampling settings. The benchmark is 1,126 problems (375 multiple choice, 751 free-response, high school to graduate level).

The repo has two stages of the project:

* `notebooks/` + `judger.py`: the **original experiments** (baseline vs. prompt engineering, 4k-token limit).
* `pipeline/`: a **rebuilt, stricter pipeline** (16k-token limit, held-out test set, one grader for every run). This is where the headline results below come from.

## Results (rebuilt pipeline)

Questions are split once with a fixed seed: 751 train, 125 dev, 250 test. Test is never trained on or used for decisions. Every row uses the same prompts, the same sampling settings (temperature 0.6, top-p 0.95, top-k 20) and the same grader; the baseline uses the original starter prompts.

| Run | Token limit | Dev (125) | **Test (250)** | MCQ | Single answer | Multi-blank | Cut off |
|---|---|---|---|---|---|---|---|
| Baseline (starter prompts) | 8k | 51.2% | 46.4% | 34.9% | 65.3% | 41.3% | 38.8% |
| Baseline (starter prompts) | 16k | 64.0% | 64.5% | 73.2% | 72.3% | 50.3% | 6.7% |
| PE | 16k | 77.6% | **71.2%** | 71.1% | 77.3% | 66.3% | 7.2% |
| PE + SFT | 16k | 76.8% | **71.6%** | 71.1% | 76.0% | 68.5% | 5.2% |

How to read this:

* **Most of the jump from the old baseline is the token limit.** At 8k tokens, 39% of baseline answers were cut off before a final answer; going to 16k takes the baseline from 46.4% to 64.5%. That is why the fair comparison is the 16k baseline row.
* **Prompt engineering adds about +6.7 points on test** over the 16k baseline (+13.6 on dev). Nearly all of it is on multi-blank questions (+16 points), where the prompts tell the model to count the `[ANS]` blanks and answer each one. Multiple choice did not improve.
* **SFT made no measurable difference** (+0.4 on test, 95% CI −2.8 to +3.6; −0.8 on dev). It trimmed cut-off answers slightly (7.2% → 5.2%). The training set (about 2,200 self-generated correct solutions plus 1,500 distilled ones) is small for a 4B model that already reasons well.
* **Caveats:** PE and PE + SFT are single samples, so a 250-question score moves a few points from sampling luck alone; the 16k baseline row averages 4 samples. Notebook 3 re-answers the test set 3 more times for every model so the final table averages 4. RL has not been run yet.

### Original experiments (4k tokens)

Scored with `judger.py` on all 1,126 problems:

| Notebook | MCQ | Free-response | Overall |
|---|---|---|---|
| `01_baseline.ipynb` | 35.20% | 49.53% | 44.76% |
| `02_prompt_engineering.ipynb` | 39.20% | 51.26% | **47.25%** |

That is +5.6% relative. The diagnosis behind it: about 80% of baseline MCQ responses (and 38% of free-response) hit the token limit before writing a final `\boxed{}` answer, so the prompts ask for brief reasoning and a final boxed answer.

## Approach (rebuilt pipeline)

* **Prompt engineering:** three system prompts (multiple choice, free-response, and questions with embedded A/B/C choices). All require a final `\boxed{}`; the free-response prompt counts `[ANS]` blanks.
* **SFT (LoRA, rank 32, all linear layers):** trained on the model's own correct, finished solutions to training questions (rejection sampling, 4 tries each, over the training questions plus 593 MMLU-Pro 10-option MCQs) and 1,500 verified DeepSeek-R1 solutions from OpenR1-Math-220k. Loss on the answer tokens only; training and test questions are never mixed.
* **GRPO (second LoRA on the merged SFT model):** the reward is the evaluation grader itself (+1 correct, partial credit on multi-blank, 0 wrong, −0.5 for an answer that is cut off or has no final answer). Prompts are chosen where the model is right some of the time but not always, so each group of 8 answers has a learning signal. `dr_grpo` loss, no KL term, 16k-token rollouts (the same budget as evaluation).
* **Grader:** SymPy-based (`pipeline/mathrl/common.py`): numeric tolerance, tuples, intervals, letters, multi-blank. Checked against all 1,126 baseline responses.
* **Honest evaluation:** paired bootstrap confidence intervals between runs, multi-sample averaging, near-duplicate filtering of external training data, and dev-set gates before test is scored.

## Hardware

Google Colab, NVIDIA A100 80GB (the 16k-token SFT and RL runs do not fit in 40GB). vLLM for generation; the notebooks save every stage to Drive and resume after a disconnect. Estimated cost in Colab compute units: notebook 0 ≈ 8–10, notebook 1 ≈ 4, notebook 2 ≈ 50–57, notebook 3 ≈ 87–92.

## Reproduce

Put the question file at `data/public.jsonl` (one JSON object per line with `id`, `question`, `answer`, and `options` for multiple choice; not included in this repo). In Colab, run the notebooks in `pipeline/notebooks/` top to bottom, in order. Each one installs pinned package versions, writes the shared `mathrl/` code, and needs no edits.

| Notebook | What it runs |
|---|---|
| `0_Baseline_16k.ipynb` | starter prompts at 16k tokens (the fair baseline) |
| `1_PE.ipynb` | engineered prompts |
| `2_PE_SFT.ipynb` | PE + SFT |
| `3_PE_SFT_RL.ipynb` | PE + SFT + GRPO, then the final 4-sample results table |

The original notebooks (`notebooks/`) expect `judger.py` in the same folder.

## Repository Structure

```
judger.py                  answer checker used for the original experiments
notebooks/                 original baseline and prompt-engineering notebooks
pipeline/
  mathrl/                  prompts, split, grader, vLLM eval, SFT, GRPO, results report
  notebooks/               the four Colab notebooks (generated from mathrl/)
  tests/                   grader and pipeline tests (CPU)
  build_notebooks.py       regenerates the notebooks from mathrl/
```

`judger.py` is a rewrite of the checker used for the original runs, so re-scoring those may differ by about a point.
