"""Re-measure the RL prompts with the SFT model (the model RL actually starts from).

build_grpo picks prompts from the *base* model's pass rates. SFT moves those: some become always-right
(a group of 8 identical rewards teaches nothing), some stay always-wrong. Here the merged SFT model answers
each candidate K times at the RL temperature and budget, and we keep the prompts it gets right sometimes but
not always (plus, when cut-offs are penalized, always-right ones whose answers run long).
Resumable; stops cleanly at --max_hours (unchecked prompts are then kept as they were).
"""
import argparse, json, os, time
from mathrl.common import (BASE_MODEL, TimeBudget, load_jsonl, append_jsonl, done_keys, reward_for)


def select(rows, checks, k, keep_long, max_tokens, min_keep):
    """rows: prompt rows from build_grpo; checks: {id: {"n_correct", "min_correct_tokens"}}."""
    keep, stats = [], {"unchecked": 0, "mixed": 0, "long": 0, "all_right": 0, "all_wrong": 0}
    for r in rows:
        c = checks.get(json.loads(r["item"])["id"])
        if c is None:
            keep.append(r); stats["unchecked"] += 1
        elif 0 < c["n_correct"] < k:
            keep.append(r); stats["mixed"] += 1
        elif c["n_correct"] == k and keep_long and (c["min_correct_tokens"] or 0) > 0.75 * max_tokens:
            keep.append(r); stats["long"] += 1
        elif c["n_correct"] == k:
            stats["all_right"] += 1
        else:
            stats["all_wrong"] += 1
    if len(keep) < min_keep:
        print(f"[recheck] only {len(keep)} prompts left (< {min_keep}); keeping the original set instead")
        return rows, stats
    return keep, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="merged SFT model dir")
    ap.add_argument("--prompts_in", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max_tokens", type=int, default=16384)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--keep_long", action="store_true")
    ap.add_argument("--min_keep", type=int, default=60)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--gpu_util", type=float, default=0.90)
    ap.add_argument("--max_hours", type=float, default=0)
    a = ap.parse_args()

    rows = load_jsonl(a.prompts_in)
    checks_path = a.out.replace(".jsonl", "_checks.jsonl")
    done = done_keys(checks_path)
    todo = [r for r in rows if json.loads(r["item"])["id"] not in done]
    budget = TimeBudget(checks_path + ".time.json", a.max_hours)
    print(f"[recheck] {len(rows)} prompts, {len(done)} already checked, {len(todo)} to go")

    if todo and not budget.would_exceed(0):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(BASE_MODEL)
        max_prompt = max(len(tok(r["prompt"], add_special_tokens=False)["input_ids"]) for r in todo)
        llm = LLM(model=a.model, tokenizer=BASE_MODEL, max_model_len=max_prompt + a.max_tokens + 64,
                  gpu_memory_utilization=a.gpu_util, seed=0, enable_prefix_caching=True)
        sp = SamplingParams(n=a.k, temperature=a.temperature, top_p=0.95, min_p=0.0, max_tokens=a.max_tokens)
        t0, last = time.time(), 0.0
        for s in range(0, len(todo), a.chunk):
            if s and budget.would_exceed(last):
                print(f"[recheck] STOPPED at the time limit; {len(todo) - s} prompts stay unchecked (kept as is)")
                break
            tc = time.time()
            chunk = todo[s:s + a.chunk]
            outs = llm.generate([r["prompt"] for r in chunk], sp)
            res = []
            for r, o in zip(chunk, outs):
                item = json.loads(r["item"])
                ok = [(reward_for(c.text, item, c.finish_reason == "length") == 1.0, len(c.token_ids)) for c in o.outputs]
                res.append({"id": item["id"], "n_correct": sum(x for x, _ in ok),
                            "min_correct_tokens": min((n for x, n in ok if x), default=None)})
            append_jsonl(checks_path, res)
            last = time.time() - tc
            budget.save()
            el = (time.time() - t0) / 60
            print(f"[recheck] {s + len(chunk)}/{len(todo)}  {el:.1f} min (~{el / (s + len(chunk)) * (len(todo) - s - len(chunk)):.0f} min left)", flush=True)

    checks = {c["id"]: c for c in load_jsonl(checks_path)} if os.path.exists(checks_path) else {}
    keep, stats = select(rows, checks, a.k, a.keep_long, a.max_tokens, a.min_keep)
    with open(a.out, "w") as f:
        for r in keep:
            f.write(json.dumps(r) + "\n")
    print(f"[recheck] kept {len(keep)}/{len(rows)} prompts  {json.dumps(stats)}")


if __name__ == "__main__":
    main()
