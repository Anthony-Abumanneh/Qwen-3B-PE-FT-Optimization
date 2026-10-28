"""Pick RL prompts where the reward will actually vary (GRPO learns nothing from all-right/all-wrong groups).

  learnable : solved by 1..K-1 of the K base-model samples
  too long  : solved every time, but answers ran long -> RL can teach finishing inside the budget
  shuffled  : learnable MCQs again with options re-ordered (new correct letter) -> no letter memorization
"""
import argparse, json, random
from mathrl.common import BASE_MODEL, LETTERS, load_jsonl, prompt_text, qtype


def shuffle_mcq(p, rng):
    perm = list(range(len(p["options"])))
    while perm == sorted(perm):
        rng.shuffle(perm)
    gold_idx = LETTERS.index(str(p["answer"]).strip().upper())
    return dict(p, id=f"{p['id']}_shuf", options=[p["options"][i] for i in perm], answer=LETTERS[perm.index(gold_idx)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--problems", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_prompt", type=int, default=2048)
    ap.add_argument("--max_completion", type=int, default=8192)
    ap.add_argument("--long_n", type=int, default=80)
    ap.add_argument("--require_fit", action="store_true",
                    help="keep a learnable problem only if one of its correct samples fits in --max_completion "
                         "(use with train_grpo --cutoff ignore, where longer rollouts teach nothing)")
    ap.add_argument("--mmlu_share", type=float, default=0.35, help="max share of RL prompts from MMLU-Pro (keeps RL focused on our own questions)")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    problems = {p["id"]: p for p in load_jsonl(a.problems)}
    rng = random.Random(0)
    learn, long_, n_nofit = [], [], 0
    for r in load_jsonl(a.samples):
        p, k, c = problems[r["id"]], r["k"], r["n_correct"]
        if 0 < c < k:
            if a.require_fit and not any(s["correct"] and s["n_tokens"] < a.max_completion for s in r["samples"]):
                n_nofit += 1
                continue
            learn.append(p)
        elif c == k and min(s["n_tokens"] for s in r["samples"]) > 0.75 * a.max_completion:
            long_.append(p)
    rng.shuffle(long_)
    longest = max((s["n_tokens"] for r in load_jsonl(a.samples) for s in r["samples"]), default=0)
    if longest < 0.9 * a.max_completion:
        print(f"[grpo-data] NOTE: samples were generated with at most ~{longest} tokens, below the RL budget of {a.max_completion}. "
              "Problems that need more than that to solve are missing from the RL set; for 16k RL set SAMPLE_MAX_TOKENS = 16384 "
              "in notebook 2 before sampling.")
    chosen = learn + long_[:a.long_n] + [shuffle_mcq(p, rng) for p in learn if p.get("options")]
    own = [p for p in chosen if p.get("src") != "mmlupro"]
    ext = [p for p in chosen if p.get("src") == "mmlupro"]
    rng.shuffle(ext)
    cap = int(len(own) * a.mmlu_share / (1 - a.mmlu_share))
    n_ext_dropped = max(0, len(ext) - cap)
    chosen = own + ext[:cap]
    rows, skipped = [], 0
    for p in chosen:
        text = prompt_text(tok, p)
        if len(tok(text, add_special_tokens=False)["input_ids"]) > a.max_prompt:
            skipped += 1
            continue
        item = {k: p[k] for k in ("id", "question", "answer") if k in p}
        if p.get("options"):
            item["options"] = p["options"]
        rows.append({"prompt": text, "item": json.dumps(item), "qtype": qtype(p)})
    rng.shuffle(rows)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[grpo-data] learnable={len(learn)} (dropped {n_nofit} whose correct answers were all longer than {a.max_completion} tokens) long={min(len(long_), a.long_n)} "
          f"shuffled_mcq={sum(1 for p in learn if p.get('options'))} -> {len(rows)} prompts "
          f"({skipped} skipped: prompt > {a.max_prompt} tokens; {n_ext_dropped} MMLU-Pro prompts dropped by the {a.mmlu_share:.0%} cap)")


if __name__ == "__main__":
    main()
