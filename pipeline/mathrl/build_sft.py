"""Build the SFT dataset (token ids + loss mask), all in the exact inference format.

Sources
  1. In-domain self-distillation: the model's own *clean* correct solutions to our TRAIN problems
     (and MMLU-Pro MCQs), shortest first -> teaches correct multi-blank formatting + finishing on time.
  2. Teacher distillation: DeepSeek-R1 step-by-step solutions from open-r1/OpenR1-Math-220k, kept only if
     verified correct, complete, our grader agrees with the final \\boxed{}, and short enough.
Each example = chat-template prompt (same system prompts as eval) + completion; loss only on the completion.
"""
import argparse, json, random, statistics
from mathrl.common import (BASE_MODEL, load_jsonl, prompt_text, grade, final_section, last_boxed, NearDupIndex)

R1_TYPES = {"Algebra", "Calculus", "Inequalities", "Number Theory", "Combinatorics", "Other"}


def openr1_examples(n, tok, max_completion, public_items, seed=0, scan_limit=60000):
    if n <= 0:
        return []
    from datasets import load_dataset
    ds = load_dataset("open-r1/OpenR1-Math-220k", "default", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=5000)
    out, scanned, seen_qt, seen_pt, n_dup = [], 0, {}, {}, 0
    dup = NearDupIndex(public_items)
    for r in ds:
        scanned += 1
        seen_qt[r.get("question_type")] = seen_qt.get(r.get("question_type"), 0) + 1
        seen_pt[r.get("problem_type")] = seen_pt.get(r.get("problem_type"), 0) + 1
        if scanned > scan_limit or len(out) >= n:
            break
        if r.get("question_type") != "math-word-problem" or r.get("problem_type") not in R1_TYPES:
            continue
        ans = str(r.get("answer") or "").strip()
        if not ans or len(ans) > 60:
            continue
        if dup.is_dup(r["problem"]):
            n_dup += 1
            continue
        gens = r.get("generations") or []
        ver = r.get("correctness_math_verify") or []
        comp = r.get("is_reasoning_complete") or [True] * len(gens)
        item = {"id": f"openr1_{r.get('uuid', scanned)}", "question": r["problem"], "answer": [ans]}
        best = None
        for g, ok, done in zip(gens, ver, comp):
            if not (ok and done) or not g.startswith("<think>") or "</think>" not in g:
                continue
            completion = g[len("<think>"):]
            completion = completion[1:] if completion.startswith("\n") else completion   # prompt already ends '<think>\n'
            fin = final_section(completion)
            if fin is None or last_boxed(fin) is None or not grade(fin, item)[0]:
                continue  # our grader must agree, so SFT never teaches a format we score as wrong
            if best is None or len(completion) < len(best):
                best = completion
        if best is None:
            continue
        ids = tok(best, add_special_tokens=False)["input_ids"]
        if len(ids) > max_completion:
            continue
        out.append({"source": "openr1", "id": item["id"], "prompt": prompt_text(tok, item), "completion_ids": ids})
    print(f"[sft] OpenR1: kept {len(out)} after scanning {scanned} rows ({n_dup} skipped as near-copies of our questions)")
    if len(out) < 0.25 * n:
        print(f"[sft] WARNING: few OpenR1 examples kept. question_type seen: {seen_qt}  problem_type seen: {seen_pt}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--problems", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_problem", type=int, default=2, help="max solutions kept per in-domain problem")
    ap.add_argument("--indomain_repeat", type=int, default=2, help="epochs over in-domain data")
    ap.add_argument("--openr1_n", type=int, default=1500)
    ap.add_argument("--max_completion", type=int, default=10240)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    problems = {p["id"]: p for p in load_jsonl(a.problems)}
    public_items = load_jsonl(a.data)

    indomain = []
    for row in load_jsonl(a.samples):
        p = problems[row["id"]]
        clean = sorted([s for s in row["samples"] if s["clean"] and s["completion_ids"]
                        and len(s["completion_ids"]) <= a.max_completion], key=lambda s: s["n_tokens"])
        seen, kept = set(), 0
        for s in clean:
            key = hash(tuple(s["completion_ids"]))  # exact duplicates only
            if key in seen:
                continue
            seen.add(key)
            indomain.append({"source": p["src"], "id": p["id"], "prompt": prompt_text(tok, p),
                             "completion_ids": s["completion_ids"]})
            kept += 1
            if kept >= a.per_problem:
                break
    n_in = len(indomain)
    ext = openr1_examples(a.openr1_n, tok, a.max_completion, public_items)
    data = indomain * a.indomain_repeat + ext
    random.Random(0).shuffle(data)
    with open(a.out, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")
    lens = [len(d["completion_ids"]) for d in data] or [0]
    stats = {"indomain_examples": n_in, "indomain_problems": len({d["id"] for d in indomain}),
             "openr1_examples": len(ext), "total_rows": len(data),
             "completion_tokens_mean": int(statistics.mean(lens)), "completion_tokens_max": max(lens),
             "total_train_tokens": sum(lens)}
    with open(a.out.replace(".jsonl", "_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print("[sft] DATASET", json.dumps(stats))


if __name__ == "__main__":
    main()
