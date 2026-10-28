"""Rejection sampling: the base model answers each training problem K times.

One pass gives us both
  * SFT data  - the model's own correct, finished solutions (exact token ids kept)
  * RL data   - per-problem pass rates, used to keep problems that are neither always nor never solved
Problems = our TRAIN split (never dev/test) + optional MMLU-Pro 10-option MCQs (same format as our MCQs).
Resumable: rerun skips problems already in --out.
--max_tokens should equal the SFT length cap: a solution longer than that can't be used for SFT,
and GRPO rolls out with an even smaller budget, so generating past it only burns compute.
"""
import argparse, json, os, random, time
from mathrl.common import (TimeBudget, BASE_MODEL, EVAL_MAX_TOKENS, EVAL_SAMPLING, load_jsonl, make_split, prompt_text,
                           qtype, gold_list, grade, final_section, last_boxed, done_keys, append_jsonl, NearDupIndex)

MMLU_CATS = {"math", "physics", "engineering", "business"}


def mmlu_items(n, public_items, seed=0):
    if n <= 0:
        return []
    from datasets import load_dataset
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    dup = NearDupIndex(public_items)
    pool, n_dup = [], 0
    for r in ds:
        if str(r["category"]).lower() not in MMLU_CATS:
            continue
        opts = [o for o in r["options"] if str(o).strip() and str(o).strip() != "N/A"]
        if len(opts) != len(r["options"]):
            continue  # keep only clean 10-option items
        if dup.is_dup(r["question"], r["options"]):
            n_dup += 1
            continue  # near-copy of one of our questions: never train on it
        pool.append({"id": f"mmlupro_{r['question_id']}", "question": r["question"], "options": list(r["options"]),
                     "answer": r["answer"], "src": "mmlupro"})
    print(f"[gen] MMLU-Pro: {len(pool)} usable, {n_dup} skipped as near-copies of our questions")
    random.Random(seed).shuffle(pool)
    return pool[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True, help="samples.jsonl (one row per problem)")
    ap.add_argument("--items_out", required=True, help="problems.jsonl (all problems incl. external)")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--mmlu_n", type=int, default=800)
    ap.add_argument("--max_tokens", type=int, default=EVAL_MAX_TOKENS)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--gpu_util", type=float, default=0.90)
    ap.add_argument("--max_hours", type=float, default=0, help="stop cleanly before this much GPU time (total over resumes); 0 = no limit")
    a = ap.parse_args()

    public = load_jsonl(a.data)
    split = make_split(public)
    saved = load_jsonl(a.items_out) if os.path.exists(a.items_out) else []
    if not saved or sum(p["src"] == "mmlupro" for p in saved) < a.mmlu_n:
        # MMLU-Pro selection is a deterministic shuffle, so a larger MMLU_N extends the saved list
        train = [dict(it, src="train") for it in public if split[it["id"]] == "train"]
        saved = train + mmlu_items(a.mmlu_n, public)
        with open(a.items_out, "w") as f:
            for p in saved:
                f.write(json.dumps(p) + "\n")
    # a smaller MMLU_N on a resumed run just stops early (the problems file keeps every item already sampled)
    mm = [p for p in saved if p["src"] == "mmlupro"][:max(a.mmlu_n, 0)]
    problems = [p for p in saved if p["src"] != "mmlupro"] + mm
    assert not any(split.get(p["id"]) in ("dev", "test") for p in problems), "dev/test leaked into training problems"
    done = done_keys(a.out)
    todo = [p for p in problems if p["id"] not in done]
    print(f"[gen] problems={len(problems)} (train={sum(p['src']=='train' for p in problems)}, "
          f"mmlupro={sum(p['src']=='mmlupro' for p in problems)}) done={len(done)} todo={len(todo)} k={a.k}")
    if not todo:
        return
    if a.max_hours and TimeBudget(a.out + ".time.json", a.max_hours).would_exceed(0):
        print(f"[gen] time limit of {a.max_hours} h already used up; keeping the {len(done)} sampled problems. "
              "Raise the limit to sample more.")
        return

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    eos_ids = {tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
    prompts = {p["id"]: prompt_text(tok, p) for p in todo}
    max_prompt = max(len(tok(t, add_special_tokens=False)["input_ids"]) for t in prompts.values())
    llm = LLM(model=BASE_MODEL, max_model_len=max_prompt + a.max_tokens + 64, gpu_memory_utilization=a.gpu_util,
              seed=0, enable_prefix_caching=True)
    sp = SamplingParams(n=a.k, max_tokens=a.max_tokens, **EVAL_SAMPLING)
    t0 = time.time()
    budget = TimeBudget(a.out + ".time.json", a.max_hours)
    last_chunk_s = 0.0
    for s in range(0, len(todo), a.chunk):
        if s and budget.would_exceed(last_chunk_s):
            print(f"[gen] STOPPED at the time limit ({a.max_hours} h of GPU time used). {len(done) + s} problems have samples; "
                  f"{len(todo) - s} were skipped (train questions go first, so these are extra MMLU-Pro ones). "
                  "Continue with the next step, or raise the limit and re-run to sample more.", flush=True)
            break
        tc = time.time()
        chunk = todo[s:s + a.chunk]
        outs = llm.generate([prompts[p["id"]] for p in chunk], sp)
        rows = []
        for p, o in zip(chunk, outs):
            samples = []
            for c in o.outputs:
                ok, frac = grade(c.text, p)
                fin = final_section(c.text)
                # "clean" = finished thinking, single final answer after </think>, and that answer is correct
                clean = (c.finish_reason == "stop" and fin is not None and last_boxed(fin) is not None
                         and grade(fin, p)[0])
                ids = [t for t in c.token_ids]
                while ids and ids[-1] in eos_ids:
                    ids.pop()
                samples.append({"correct": ok, "frac": frac, "clean": clean, "n_tokens": len(c.token_ids),
                                "finish_reason": c.finish_reason,
                                "completion_ids": ids if clean else None,
                                "text": c.text if clean else c.text[-1500:]})
            rows.append({"id": p["id"], "src": p["src"], "qtype": qtype(p), "k": a.k,
                         "n_correct": sum(x["correct"] for x in samples), "samples": samples})
        append_jsonl(a.out, rows)
        last_chunk_s = time.time() - tc
        budget.save()
        el = (time.time() - t0) / 60
        print(f"[gen] {s + len(chunk)}/{len(todo)} problems  {el:.1f} min  "
              f"(~{el / (s + len(chunk)) * (len(todo) - s - len(chunk)):.0f} min left)", flush=True)


if __name__ == "__main__":
    main()
