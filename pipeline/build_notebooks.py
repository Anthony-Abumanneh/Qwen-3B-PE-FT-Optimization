"""Generate the three Colab notebooks. Each one embeds the identical, tested mathrl/ package."""
import json, os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "notebooks")
os.makedirs(OUT, exist_ok=True)
FILES = ["mathrl/__init__.py", "mathrl/common.py", "mathrl/run_eval.py", "mathrl/gen_samples.py",
         "mathrl/build_sft.py", "mathrl/train_sft.py", "mathrl/merge_lora.py", "mathrl/build_grpo.py",
         "mathrl/train_grpo.py", "mathrl/recheck_grpo.py", "mathrl/report.py", "tests/test_grader.py"]


def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(True)}
def code(src): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                       "source": src.strip("\n").splitlines(True)}


SETUP_SETTINGS = r'''
#@title Settings (edit here only)
USE_DRIVE = True          # keep True: every stage saves to Drive and resumes after a disconnect
WORK = "/content/drive/MyDrive/qwen_math_ft" if USE_DRIVE else "/content/qwen_math_ft"
CODE = "/content/qwen_ft"
{extra}
import os, sys, subprocess, json, shutil
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
for d in [WORK, f"{{WORK}}/data", f"{{WORK}}/results", f"{{WORK}}/sft", f"{{WORK}}/grpo", f"{{CODE}}/mathrl", f"{{CODE}}/tests"]:
    os.makedirs(d, exist_ok=True)
DATA = f"{{WORK}}/data/public.jsonl"
BASELINE = f"{{WORK}}/data/starter_results_backup.jsonl"   # optional: old baseline responses, rescored for the table
# Data files are read from Drive (Colab's upload button fails on files of a few MB).
# Looks in MyDrive/qwen_math_ft/data/, then MyDrive/qwen_math_ft/, MyDrive/, and /content/ (the Files panel).
for name, dst in [("public.jsonl", DATA), ("starter_results_backup.jsonl", BASELINE)]:
    if not os.path.exists(dst):
        for src in [f"{{WORK}}/{{name}}", f"{{os.path.dirname(WORK)}}/{{name}}", f"/content/{{name}}"]:
            if os.path.exists(src):
                shutil.copy(src, dst); print(f"copied {{src}} -> {{dst}}"); break
if not os.path.exists(DATA):
    raise FileNotFoundError(
        "public.jsonl not found. Put it (and optionally starter_results_backup.jsonl) in Google Drive at\n"
        f"    {{WORK}}/data/   (i.e. My Drive > qwen_math_ft > data)\n"
        "or drag it into the Files panel on the left (it lands in /content), then re-run this cell.")
print("data:", DATA, "| baseline file:", "found" if os.path.exists(BASELINE) else "not found (baseline row will be skipped)")
os.environ.update(PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")

class StepFailed(RuntimeError):
    def __init__(self, tail):
        last = "".join(tail.splitlines(True)[-25:])
        super().__init__("Step failed. Last lines of its log:\n\n" + last + "\nRe-running this cell resumes where it stopped.")
        self.oom = "out of memory" in tail.lower()

def run(cmd, done_if=None):
    """Run one pipeline step as its own process (frees all GPU memory when it ends), streaming its log."""
    if done_if and os.path.exists(done_if):
        print(f"[skip] already done -> {{done_if}}"); return
    print("$", cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, cwd=CODE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail = []
    for line in p.stdout:
        print(line, end="")
        tail.append(line)
        if len(tail) > 200: del tail[:100]
    if p.wait() != 0:
        raise StepFailed("".join(tail))

def summary(path):
    s = path.replace(".jsonl", "_summary.json")
    return json.load(open(s)) if os.path.exists(s) else None

def gpu_units_per_hour():
    """Colab compute units per hour for this GPU. Google doesn't publish these; third-party measurements, Mar 2026."""
    try:
        q = subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits", shell=True,
                           capture_output=True, text=True).stdout.strip().splitlines()[0]
        name, mem = q.rsplit(",", 1); mem = float(mem)
    except Exception:
        return None, "no GPU"
    if "A100" in name: return (7.52 if mem > 60000 else 5.40), name.strip()
    if "L4" in name: return 1.71, name.strip()
    if "T4" in name: return 1.19, name.strip()
    return None, name.strip()
RATE, GPU_NAME = gpu_units_per_hour()

def hours_for(units):
    """Turn a compute-unit limit into GPU hours on this GPU (unknown GPU -> assume the pricier A100 80GB rate)."""
    return round(units / (RATE or 7.52), 2)
print("WORK =", WORK, "| GPU:", GPU_NAME, f"(~{{RATE}} units/h)" if RATE else "(rate unknown)")
'''

INSTALL = r'''
#@title Install pinned packages (~5-8 min, once per runtime)
# These exact versions were checked against each other's requirements and support Colab's Python 3.13:
#   vllm 0.15.1 needs torch 2.9.1 + transformers <5; unsloth 2026.5.8 needs peft>=0.18, trl<=0.24, torch<2.11.
#   peft is pinned to 0.18.1 on purpose: peft>=0.19 imports EmbeddingParallel from transformers, which
#   breaks checkpoint resume with this transformers version (the error from last time).
#   torchao is pinned to 0.15.0, the release built for torch 2.9.1 (0.17+ needs torch 2.11 and breaks vLLM's import).
#   Every package is pinned, so a new release can't change the environment between runs.
# Any failure here stops "Run all" and prints the real error.
import subprocess, sys
def sh(cmd):
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    out = (r.stdout + r.stderr).strip()
    if r.returncode != 0:
        print(out[-6000:])
        raise RuntimeError(f"Install step failed (exit code {r.returncode}). The error is printed just above.")
    return out
print(sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"))
sh('pip install -q "torch==2.9.1" "vllm==0.15.1" "xformers==0.0.33.post2" "transformers==4.57.6" "trl==0.22.2" '
   '"peft==0.18.1" "unsloth==2026.5.8" "unsloth_zoo==2026.5.4" "datasets==4.3.0" "accelerate==1.15.0" '
   '"bitsandbytes==0.50.2" "sympy==1.14.0" "torchao==0.15.0"')
# Verify in a separate process (this notebook's own Python never imports torch, so no restart is needed).
VERIFY = (
    "import unsloth\n"
    "import torch, vllm, transformers, trl, peft, datasets, torchao\n"
    "from vllm import LLM, SamplingParams\n"
    "import transformers.quantizers.quantizer_torchao\n"
    "v = {'torch': torch.__version__.split('+')[0], 'vllm': vllm.__version__, 'transformers': transformers.__version__,\n"
    "     'trl': trl.__version__, 'peft': peft.__version__, 'datasets': datasets.__version__, 'torchao': torchao.__version__}\n"
    "exp = {'torch': '2.9.1', 'vllm': '0.15.1', 'transformers': '4.57.6', 'trl': '0.22.2', 'peft': '0.18.1', 'datasets': '4.3.0', 'torchao': '0.15.0'}\n"
    "bad = {k: (v[k], exp[k]) for k in exp if v[k] != exp[k]}\n"
    "assert not bad, f'version mismatch {bad}'\n"
    "assert torch.cuda.is_available(), 'No GPU - Runtime > Change runtime type > A100'\n"
    "assert torch.cuda.is_bf16_supported(), 'This GPU has no bfloat16 (T4/V100). Use an A100.'\n"
    "print(v, '| GPU:', torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory / 1e9), 'GB')\n")
open("/tmp/verify_env.py", "w").write(VERIFY)
print(sh(f"{sys.executable} /tmp/verify_env.py"))
print("install OK")
'''


def write_cells():
    cells = [md("## Project code\nThe same files are written by all three notebooks, so prompts, data split and grader are identical. Run these cells; no need to edit.")]
    for f in FILES:
        src = open(os.path.join(ROOT, f)).read()
        cells.append(code(f"%%writefile /content/qwen_ft/{f}\n{src}"))
    cells.append(code('run("python tests/test_grader.py")\nrun(f"python -c \\"from mathrl.common import *; items=load_jsonl(\'{DATA}\'); s=make_split(items); import collections; print(collections.Counter(s.values()))\\"")'))
    return cells


def notebook(cells):
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100", "machine_shape": "hm"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


RUNS = '"Baseline (starter prompts, 8k)={BASELINE}" "Baseline (starter prompts, 16k)={WORK}/results/base16k_{S}.jsonl" "PE={WORK}/results/pe_{S}.jsonl" "PE + SFT={WORK}/results/sft_{S}.jsonl" "PE + SFT + RL={WORK}/results/rl_{S}.jsonl"'
REPORT = r'''
#@title Results table (same grader, same questions for every row)
for S in ["dev", "test"]:
    run(f'python -m mathrl.report --data {DATA} --split {S} --runs ''' + RUNS + r'''')
'''

# ----------------------------------------------------------------------------------------------
nb0 = [md(r'''
# 0 · Baseline at 16k (fair comparison)
Re-runs the **original baseline**, with the starter prompts copied verbatim from the baseline notebook, under the *same* protocol as every other row:
**16,384 max tokens**, temperature 0.6 / top-p 0.95 / top-k 20, same model, same grader, same 125 dev / 250 test questions.
The only difference from notebook 1 (PE) is the system prompts, so **Baseline 16k → PE** is the honest effect of prompt engineering, and **Baseline 8k → Baseline 16k** is the effect of the longer token budget.

Test is answered **4 times** per question (seeds 0–3), like the final PE / SFT / RL rows in notebook 3, so every row averages the same way.

**Before running:** Runtime → Change runtime type → **A100**. Data files in **My Drive › qwen_math_ft › data** (already there from notebook 1). Then Runtime → Run all; nothing to edit.
**Cost:** ~1–1.5 h ≈ 8–10 units. Resumable: a disconnect only means Run all again. Can run any time, before or after notebooks 2 and 3.
'''),
       code(SETUP_SETTINGS.format(extra="")), code(INSTALL)] + write_cells() + [
       md("## Baseline at 16k on dev (125) and test (250 × 4 samples)"),
       code('run(f"python -m mathrl.run_eval --data {DATA} --split dev --prompts starter --label \'Baseline 16k\' --out {WORK}/results/base16k_dev.jsonl", done_if=f"{WORK}/results/base16k_dev_summary.json")'),
       code('''run(f"python -m mathrl.run_eval --data {DATA} --split test --prompts starter --label 'Baseline 16k' --out {WORK}/results/base16k_test.jsonl", done_if=f"{WORK}/results/base16k_test_summary.json")
for i in range(1, 4):
    out = f"{WORK}/results/base16k_test_s{i}.jsonl"
    run(f"python -m mathrl.run_eval --data {DATA} --split test --prompts starter --seed {i} --label 'Baseline 16k sample {i}' --out {out}", done_if=out.replace(".jsonl", "_summary.json"))'''),
       code(REPORT),
       md(r"""
The test rows say *(avg of N samples)*: until notebook 3 adds the extra PE / SFT / RL samples, PE is still a single sample, so compare **Baseline 16k → PE** again after notebook 3 for the cleanest number.
"""),]

nb1 = [md(r'''
# 1 · Prompt engineering (PE) under the new protocol
Re-runs your prompt-engineered model with **16,384 max tokens** (Qwen's card recommends ≥32k; your baseline lost ~38% of answers to the 8k cut-off).
This gives the **PE** row that SFT and RL are compared against.

**Frozen protocol for every row:** Qwen3-4B-Thinking-2507 · your 3 PE system prompts (verbatim) · temperature 0.6 / top-p 0.95 / top-k 20 · 16,384 tokens · one sample · same grader.
**Data split (fixed seed, stratified):** 250 test · 125 dev · 751 train. Test is only scored, never trained on or used for decisions.

**Before running:** Runtime → Change runtime type → **A100** (80GB if offered; 40GB is fine for this notebook). Put `public.jsonl` (and optionally `starter_results_backup.jsonl` for the baseline row) in Google Drive under **My Drive › qwen_math_ft › data** (or drag them into the Files panel on the left).
**Cost:** ~30–45 min ≈ 4 compute units. Nothing to edit: just Runtime → Run all.
**When it finishes:** Runtime → *Disconnect and delete runtime*. Colab bills while the runtime is connected, even when idle.
'''),
       code(SETUP_SETTINGS.format(extra="")), code(INSTALL)] + write_cells() + [
       md("## Evaluate PE on dev (125) and test (250)\nResumable: if Colab disconnects, re-run the cell and it continues from the last saved chunk."),
       code('run(f"python -m mathrl.run_eval --data {DATA} --split dev --label PE --out {WORK}/results/pe_dev.jsonl", done_if=f"{WORK}/results/pe_dev_summary.json")'),
       code('run(f"python -m mathrl.run_eval --data {DATA} --split test --label PE --out {WORK}/results/pe_test.jsonl", done_if=f"{WORK}/results/pe_test_summary.json")'),
       code(REPORT)]

# ----------------------------------------------------------------------------------------------
nb2 = [md(r'''
# 2 · PE + SFT
Same frozen protocol as notebook 1; the **only** change is a LoRA adapter trained with SFT.

**What the SFT data is (all verified correct, all in the exact inference format):**
1. **Self-distillation (in-domain):** the base model answers each of our 751 *train* questions (+800 MMLU-Pro 10-option MCQs, the same format as your MCQs, which are the weakest category) 4 times; we keep its correct, finished answers, shortest first.
   This targets the two biggest measured failures: answers cut off before `\boxed{}` (MCQ only finished 130/375 at 8k) and multi-blank answers with the wrong *number* of values (41% of finished multi-blank answers).
2. **Teacher distillation:** up to 1,500 step-by-step solutions written by **DeepSeek-R1** (a much stronger model) from OpenR1-Math-220k, kept only if verified correct, complete, our grader agrees, ≤16k tokens, and not a near-copy of any of your questions.

**Training:** LoRA r=32 on all linear layers of Qwen3-4B-Thinking-2507 (bf16), loss only on the answer tokens, lr 1e-4 cosine, effective batch 16, 2 passes over in-domain data + 1 over R1 data.
**Safety gate:** we score dev first; if SFT is not at least as good as PE on dev, the notebook tells you exactly what to change before you spend more units.

**Run on an A100 80GB** (~7.5 units/h). Nothing to edit: Runtime → Run all. Every step saves to Drive and resumes if Colab disconnects.
**Cost:** sampling ≤ 40 units (≈ 5.3 h; stops cleanly at the limit, train questions first) · build ~15 min · train ~1.5–2 h · eval ~45 min → **≈ 50–57 units**.
If training runs out of GPU memory on a 16k-token solution, it automatically retries with solutions up to 10k tokens.
'''),
       code(SETUP_SETTINGS.format(extra='# The plan is fixed; nothing here needs editing.\nK = 4                 # answers sampled per training question\nMMLU_N = 800          # extra 10-option MCQs from MMLU-Pro\nOPENR1_N = 1500       # DeepSeek-R1 solutions\nSFT_LR = 1e-4\nSFT_MAX_COMPLETION = 16384   # longest solution used for SFT (same budget as eval)\nSAMPLE_MAX_TOKENS = 16384    # sampling budget (same as eval and RL)\nSAMPLING_UNITS = 40          # compute-unit limit for step 1')), code(INSTALL)] + write_cells() + [
       md("## Step 1 · Sample solutions from the base model (longest step, resumable)"),
       code('run(f"python -m mathrl.gen_samples --data {DATA} --out {WORK}/sft/samples.jsonl --items_out {WORK}/sft/problems.jsonl --k {K} --mmlu_n {MMLU_N} --max_tokens {SAMPLE_MAX_TOKENS} --max_hours {hours_for(SAMPLING_UNITS)}")'),
       md("## Step 2 · Build the SFT dataset"),
       code('run(f"python -m mathrl.build_sft --samples {WORK}/sft/samples.jsonl --problems {WORK}/sft/problems.jsonl --data {DATA} --out {WORK}/sft/sft_data.jsonl --openr1_n {OPENR1_N} --max_completion {SFT_MAX_COMPLETION}", done_if=f"{WORK}/sft/sft_data_stats.json")\nprint(open(f"{WORK}/sft/sft_data_stats.json").read())'),
       md("## Step 3 · Train the LoRA adapter (checkpoints every 25 steps; re-run resumes)"),
       code(r'''
FALLBACK = f"{WORK}/sft/use_10k"     # written if 16k-token solutions don't fit in GPU memory
ADAPTER = f"{WORK}/sft/adapter/adapter_model.safetensors"
if not os.path.exists(FALLBACK):
    try:
        run(f"python -m mathrl.train_sft --sft_data {WORK}/sft/sft_data.jsonl --ckpt_dir {WORK}/sft/ckpt --adapter_out {WORK}/sft/adapter --lr {SFT_LR} --max_seq {SFT_MAX_COMPLETION + 2048}", done_if=ADAPTER)
    except StepFailed as e:
        if not e.oom: raise
        print("\n>>> Out of GPU memory with 16k-token solutions: retrying automatically with solutions up to 10k tokens.\n")
        open(FALLBACK, "w").close()
if os.path.exists(FALLBACK):
    run(f"python -m mathrl.build_sft --samples {WORK}/sft/samples.jsonl --problems {WORK}/sft/problems.jsonl --data {DATA} --out {WORK}/sft/sft_data_10k.jsonl --openr1_n {OPENR1_N} --max_completion 10240", done_if=f"{WORK}/sft/sft_data_10k_stats.json")
    run(f"python -m mathrl.train_sft --sft_data {WORK}/sft/sft_data_10k.jsonl --ckpt_dir {WORK}/sft/ckpt_10k --adapter_out {WORK}/sft/adapter --lr {SFT_LR} --max_seq 12288", done_if=ADAPTER)
'''),
       md("## Step 4 · Dev gate, then test"),
       code(r'''
run(f"python -m mathrl.run_eval --data {DATA} --split dev --label 'PE + SFT' --lora {WORK}/sft/adapter --out {WORK}/results/sft_dev.jsonl", done_if=f"{WORK}/results/sft_dev_summary.json")
sys.path.insert(0, CODE)
from mathrl.common import load_jsonl, per_question, paired_diff
pe_dev, sft_dev = f"{WORK}/results/pe_dev.jsonl", f"{WORK}/results/sft_dev.jsonl"
if summary(pe_dev) is None:
    print("Run notebook 1 first to get the PE dev score to compare against.")
else:
    items = load_jsonl(DATA)
    pe, sft = summary(pe_dev), summary(sft_dev)
    c = paired_diff(per_question(pe_dev, items, "dev"), per_question(sft_dev, items, "dev"))
    print(f"DEV  PE {pe['overall']:.1f}%  ->  PE+SFT {sft['overall']:.1f}%   change {c['diff']:+.1f} pts (95% CI {c['lo']:+.1f} to {c['hi']:+.1f}; "
          f"fixed {c['fixed']}, broke {c['broke']})   truncated {pe['truncated_pct']:.0f}% -> {sft['truncated_pct']:.0f}%")
    if c["hi"] < 0:
        print("SFT is clearly worse than PE on dev. Before evaluating test: set SFT_LR = 5e-5 (gentler) and/or OPENR1_N = 0 (pure self-distillation),\n"
              f"delete {WORK}/sft/adapter, {WORK}/sft/ckpt and {WORK}/sft/sft_data*.jsonl/_stats.json (samples are reused), then re-run steps 2-4.")
    elif c["diff"] < 0:
        print("Slightly below PE, but within the noise of 125 dev questions. Go ahead to test; retraining on this alone would mostly chase noise.")
    else:
        print("OK - go ahead to test.")
'''),
       code('run(f"python -m mathrl.run_eval --data {DATA} --split test --label \'PE + SFT\' --lora {WORK}/sft/adapter --out {WORK}/results/sft_test.jsonl", done_if=f"{WORK}/results/sft_test_summary.json")'),
       code(REPORT),
       md(r"""
### If something goes wrong
* **Disconnected / runtime reset:** run all cells again. Every step resumes (sampling continues from the last saved chunk, training from the last checkpoint on Drive, evals from the last saved chunk).
* **Out of GPU memory during training:** handled automatically (retries with ≤10k-token solutions; a 1-step memory check runs first, so this costs minutes, not hours).
* **`[sft] WARNING: few OpenR1 examples kept`:** the dataset's labels changed; the warning prints what it saw. You can continue with `OPENR1_N = 0`.
* **`[gen] STOPPED at the time limit`:** expected if sampling is slower than estimated; the notebook just continues with what was sampled (your train questions go first).
"""),]

# ----------------------------------------------------------------------------------------------
nb3 = [md(r'''
# 3 · PE + SFT + RL (GRPO)
Starts from the SFT model of notebook 2 (needs `sft/adapter` and `sft/samples.jsonl` on Drive) and adds a second LoRA trained with GRPO. Same frozen eval protocol.

**What makes this RL stage different from last time**
| Last time | Now |
|---|---|
| reward compared the answer to the *whole solution text* (almost never matched) | reward = **the evaluation grader itself** on the final `\boxed{}` |
| Qwen3-4B-**Base** + custom `<SOLUTION>` tags | the **same Thinking model** as the baseline, same prompts, same `\boxed{}` format |
| random problems (many groups all-right or all-wrong → zero signal) | problems the model solves **1–3 of 4 times**, **re-checked with the SFT model** (the model RL starts from), + long-but-correct ones, + option-shuffled MCQs; MMLU-Pro capped at 35% so RL stays on your own question types |
| truncation ignored | rollouts get **16,384 tokens, the same budget as eval**; an answer cut off there also fails at eval, so it gets −0.5 |
| Unsloth default `bnpo` loss | `dr_grpo` (no length bias), no KL (β=0), clip-higher ε=0.28, 8 samples × 4 problems per step (a bigger, less noisy batch costs little extra because generation time is set by the longest answer), LoRA learning rate 1e-5 |

Reward: **+1** correct · **0–0.5** partial credit on multi-blank · **0** wrong · **−0.5** no final `\boxed{}` answer or cut off at 16k.
We keep the step-20 and step-40 adapters and the final one, pick the best on **dev**, and only then score **test**. Every 10 steps the log prints a `HEALTH` line (accuracy, cut-offs, groups without signal, min/step) with a warning if something looks wrong, so you can stop early instead of paying for a broken run.

**Run on an A100 80GB** (16 rollouts × 16k tokens don't fit in 40GB; the notebook checks). Nothing to edit: Runtime → Run all.
**Cost (≈ 7.5 units/h):** merge ~5 min · re-check ≤ 12 units · GRPO ≤ 55 units (estimated 6–9 min/step with 32 answers per step → roughly 45–70 steps; it prints real `min/step` and stops cleanly at the limit; constant learning rate, so stopping early is clean) · dev/test evals ~8 units · 3 extra test samples for all three models ~14 units → **≈ 87–92 units**.
The extra test samples make every row of the final table an average of 4 answers per question, which roughly halves the noise in the comparison.
'''),
       code(SETUP_SETTINGS.format(extra='# The plan is fixed; nothing here needs editing.\nGRPO_STEPS = 120\nGRPO_LR = 1e-5\nPROMPTS_PER_STEP = 4       # 4 problems x 8 answers per step\nMAX_COMPLETION = 16384     # RL rollout budget = eval budget\nCUT_OFF_ROLLOUTS = "penalize"   # a rollout cut off at 16k also fails at eval -> -0.5\nGRPO_GPU_UTIL = 0.85       # retried automatically at 0.70 if GRPO runs out of GPU memory\nRECHECK_UNITS = 12         # compute-unit limit for re-checking RL prompts with the SFT model\nGRPO_UNITS = 55            # compute-unit limit for GRPO training\nEXTRA_TEST_SAMPLES = 3     # extra answers per test question for all three models (final table = average of 4)\nMERGED = "/content/sft_merged"   # local disk; rebuilt in ~5 min if the runtime resets')), code(INSTALL)] + write_cells() + [
       code('assert os.path.exists(f"{WORK}/sft/adapter/adapter_model.safetensors") and os.path.exists(f"{WORK}/sft/samples.jsonl"), "Run notebook 2 first"\n'
            'assert RATE == 7.52, f"This notebook needs an A100 80GB (this runtime: {GPU_NAME}). Disconnect and reconnect an A100 runtime until nvidia-smi shows 80GB."\n'
            'print(f"RL: rollouts up to {MAX_COMPLETION} tokens; re-check limit {RECHECK_UNITS} units, GRPO limit {GRPO_UNITS} units = {hours_for(GRPO_UNITS)} h on {GPU_NAME}")'),
       md("## Step 1 · Merge the SFT adapter into the weights (RL starts from the SFT model)"),
       code('run(f"python -m mathrl.merge_lora --adapter {WORK}/sft/adapter --out {MERGED}", done_if=f"{MERGED}/config.json")'),
       md("## Step 2 · Pick RL prompts with real reward variance"),
       code('run(f"python -m mathrl.build_grpo --samples {WORK}/sft/samples.jsonl --problems {WORK}/sft/problems.jsonl --out {WORK}/grpo/prompts.jsonl --max_completion {MAX_COMPLETION}", done_if=f"{WORK}/grpo/prompts.jsonl")'),
       md("## Step 2b · Re-check those prompts with the SFT model\nThe prompts above were chosen from the *base* model's answers; SFT changes which ones it gets right. Keep only prompts the SFT model gets right sometimes but not always, so every RL group carries signal."),
       code('run(f"python -m mathrl.recheck_grpo --model {MERGED} --prompts_in {WORK}/grpo/prompts.jsonl --out {WORK}/grpo/prompts_checked.jsonl --max_tokens {MAX_COMPLETION} --keep_long --max_hours {hours_for(RECHECK_UNITS)}", done_if=f"{WORK}/grpo/prompts_checked.jsonl")'),
       md("## Step 3 · GRPO (checkpoints every 10 steps; re-run resumes)\nWatch the `[grpo] step N: acc=… no_answer=… cut_off=…` lines: accuracy should drift up and `cut_off` (answers that hit 16k) down. In trl's logs, `frac_reward_zero_std` near 1.0 means most groups had no signal."),
       code(r'''
grpo = lambda util: run(f"python -m mathrl.train_grpo --base {MERGED} --prompts {WORK}/grpo/prompts_checked.jsonl --ckpt_dir {WORK}/grpo/ckpt --adapter_out {WORK}/grpo/adapter --steps {GRPO_STEPS} --lr {GRPO_LR} --prompts_per_step {PROMPTS_PER_STEP} --max_completion {MAX_COMPLETION} --keep_steps 20,40 --gpu_util {util} --max_hours {hours_for(GRPO_UNITS)} --cutoff {CUT_OFF_ROLLOUTS}", done_if=f"{WORK}/grpo/adapter/adapter_model.safetensors")
try:
    grpo(GRPO_GPU_UTIL)
except StepFailed as e:
    if not e.oom: raise
    print("\n>>> Out of GPU memory: retrying automatically with less memory for generation (resumes from the last checkpoint).\n")
    grpo(0.70)
'''),
       md("## Step 4 · Choose the RL checkpoint on dev, then score test"),
       code(r'''
cands = {"step20": f"{WORK}/grpo/grpo_adapter_step20", "step40": f"{WORK}/grpo/grpo_adapter_step40", "final": f"{WORK}/grpo/adapter"}
dev = {}
for name, path in cands.items():
    if os.path.exists(f"{path}/adapter_model.safetensors"):
        out = f"{WORK}/results/rl_{name}_dev.jsonl"
        run(f"python -m mathrl.run_eval --data {DATA} --split dev --base {MERGED} --lora {path} --label 'RL {name}' --out {out}", done_if=out.replace(".jsonl", "_summary.json"))
        dev[name] = summary(out)["overall"]
best = max(dev, key=dev.get)
print("dev:", {k: f"{v:.1f}%" for k, v in dev.items()}, "| chosen:", best)
sys.path.insert(0, CODE)
from mathrl.common import load_jsonl, per_question, paired_diff
sft_dev = f"{WORK}/results/sft_dev.jsonl"
if os.path.exists(sft_dev):
    items = load_jsonl(DATA)
    c = paired_diff(per_question(sft_dev, items, "dev"), per_question(f"{WORK}/results/rl_{best}_dev.jsonl", items, "dev"))
    print(f"DEV  PE+SFT -> PE+SFT+RL: change {c['diff']:+.1f} pts (95% CI {c['lo']:+.1f} to {c['hi']:+.1f}; fixed {c['fixed']}, broke {c['broke']})")
    if c["hi"] < 0:
        print("RL is clearly worse than SFT on dev. Test is still scored so you have the number, but report PE+SFT as the main result; "
              "a retry would use GRPO_LR = 3e-6 (delete grpo/ckpt + grpo/adapter and re-run step 3).")
    elif c["diff"] < 0:
        print("Slightly below SFT, but within the noise of 125 dev questions. Go ahead to test.")
BEST_RL = cands[best]
'''),
       code('run(f"python -m mathrl.run_eval --data {DATA} --split test --base {MERGED} --lora {BEST_RL} --label \'PE + SFT + RL\' --out {WORK}/results/rl_test.jsonl", done_if=f"{WORK}/results/rl_test_summary.json")'),
       code(r'''
#@title Results table (same grader, same questions for every row)
import shutil
for S in ["dev", "test"]:
    rl = f"{WORK}/results/rl_{S}.jsonl"
    if S == "dev" and not os.path.exists(rl) and "best" in globals():
        shutil.copy(f"{WORK}/results/rl_{best}_dev.jsonl", rl)
    run(f'python -m mathrl.report --data {DATA} --split {S} --runs ''' + RUNS + r'''')
'''),
       md(r"""
## Step 5 · More precise test scores
With one answer per question, a 250-question score moves ±6 points from sampling luck alone, so a real +3 from SFT could look like noise.
This re-answers the 250 test questions 3 more times with **all three** models (PE, PE+SFT, PE+SFT+RL), each with a new seed, so every row of the final table is an average of 4 answers per question.
Same prompts, settings and grader as before; only the random seed differs. Dev and training are untouched.
"""),
       code(r'''
#@title Extra test samples + final table
for i in range(1, EXTRA_TEST_SAMPLES + 1):
    for tag, model in [("pe", ""), ("sft", f"--lora {WORK}/sft/adapter"), ("rl", f"--base {MERGED} --lora {BEST_RL}")]:
        out = f"{WORK}/results/{tag}_test_s{i}.jsonl"
        run(f"python -m mathrl.run_eval --data {DATA} --split test --seed {i} {model} --label '{tag} sample {i}' --out {out}",
            done_if=out.replace(".jsonl", "_summary.json"))
if EXTRA_TEST_SAMPLES:
    S = "test"
    run(f'python -m mathrl.report --data {DATA} --split test --runs ''' + RUNS + r'''')
'''),
       md(r"""
### If something goes wrong
* **Disconnected / runtime reset:** run all cells again; the merge is redone locally (~5 min) and GRPO resumes from its last Drive checkpoint.
* **Out of GPU memory in GRPO:** handled automatically (one retry with less generation memory, resuming from the checkpoint).
* **`[grpo] STOPPING ... limit`:** the adapter at that step is used as the final one; the dev comparison of the saved checkpoints still runs.
* **Not an A100 80GB:** Runtime → Disconnect and delete runtime, then reconnect with A100 until the first cell shows 80GB.
"""),]

for name, cells in [("0_Baseline_16k.ipynb", nb0), ("1_PE.ipynb", nb1), ("2_PE_SFT.ipynb", nb2), ("3_PE_SFT_RL.ipynb", nb3)]:
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(notebook(cells), f, indent=1)
    print("wrote", name, len(cells), "cells")
