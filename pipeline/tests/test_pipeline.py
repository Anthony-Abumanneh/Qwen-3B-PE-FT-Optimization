"""CPU dry run of the data pipeline with a fake tokenizer + fake HF datasets (no GPU, no downloads).

Checks: split integrity (no dev/test leakage), SFT example format + loss masking, OpenR1 filtering,
MMLU-Pro filtering, GRPO prompt selection, reward wrapper (truncation, partial credit).
"""
import sys, os, json, types, random, tempfile
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
DATA = sys.argv[1] if len(sys.argv) > 1 else "data/public.jsonl"

SPECIAL = {"<|im_end|>": 151645, "<|endoftext|>": 151643, "<|im_start|>": 151644}


class FakeTok:
    pad_token_id = 151643
    eos_token = "<|im_end|>"
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 150000 for c in text]}
    def convert_tokens_to_ids(self, t): return SPECIAL[t]
    def decode(self, ids): return "".join(chr(i) if i < 150000 else "<S>" for i in ids)
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
        s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
        return s + ("<|im_start|>assistant\n<think>\n" if add_generation_prompt else "")
    def save_pretrained(self, *a): pass


fake_tf = types.ModuleType("transformers")
fake_tf.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: FakeTok())
sys.modules["transformers"] = fake_tf


class FakeStream(list):
    def shuffle(self, seed=0, buffer_size=0):
        r = FakeStream(self); random.Random(seed).shuffle(r); return r


def fake_load_dataset(name, *args, **kw):
    if "MMLU-Pro" in name:
        rows = []
        for i in range(60):
            cat = ["math", "physics", "law", "business"][i % 4]
            opts = [f"opt{j}" for j in range(10)]
            if i % 10 == 7: opts[9] = "N/A"
            rows.append({"question_id": i, "question": f"mmlu q {i}", "options": opts, "answer": "C", "category": cat})
        return rows
    if "OpenR1" in name:
        rows = []
        for i in range(40):
            good = "<think>\nlet me compute\n</think>\n\nSo the answer is \\boxed{42}."
            longer = "<think>\n" + "x" * 500 + "\n</think>\n\n\\boxed{42}"
            wrongfmt = "<think>\nhmm\n</think>\n\nanswer 42"          # no box -> must be rejected
            rows.append({"problem": f"openr1 problem {i}", "answer": "42", "uuid": f"u{i}",
                         "question_type": "math-word-problem" if i % 5 else "proof",
                         "problem_type": "Geometry" if i % 7 == 0 else "Algebra",
                         "generations": [longer, good, wrongfmt] if i % 3 else [wrongfmt],
                         "correctness_math_verify": [True, True, True] if i % 3 else [True],
                         "is_reasoning_complete": [True, True, True] if i % 3 else [True]})
        return FakeStream(rows)
    raise ValueError(name)


fake_ds = types.ModuleType("datasets")
fake_ds.load_dataset = fake_load_dataset
sys.modules["datasets"] = fake_ds

from mathrl.common import load_jsonl, make_split, gold_list, qtype, grade, reward_for
from mathrl import gen_samples, build_sft, build_grpo
from mathrl.train_sft import make_rows
from mathrl.train_grpo import make_reward_fn


def fake_answer_text(it, correct, finish=True):
    golds = gold_list(it)
    if it.get("options"):
        ans = golds[0] if correct else ("A" if golds[0] != "A" else "B")
    else:
        ans = ", ".join(golds) if correct else ", ".join(["999"] * len(golds))
    body = "reasoning steps " * 3
    return (body + "</think>\n\nFinal: \\boxed{" + ans + "}") if finish else body * 50


def main():
    tok = FakeTok()
    items = load_jsonl(DATA)
    split = make_split(items)
    tmp = tempfile.mkdtemp()
    problems_path, samples_path = f"{tmp}/problems.jsonl", f"{tmp}/samples.jsonl"

    # ---- problems: train split + filtered MMLU-Pro (real function, fake dataset)
    mm = gen_samples.mmlu_items(20, items)
    assert all(m["id"].startswith("mmlupro_") and len(m["options"]) == 10 and "N/A" not in m["options"] for m in mm)
    assert all(int(m["id"].split("_")[1]) % 4 in (0, 1, 3) for m in mm), "law questions must be filtered out"
    problems = [dict(it, src="train") for it in items if split[it["id"]] == "train"] + mm
    with open(problems_path, "w") as f:
        for p in problems: f.write(json.dumps(p) + "\n")

    # ---- synthetic samples in gen_samples' exact row format (k=4, varied pass rates)
    rng = random.Random(0)
    with open(samples_path, "w") as f:
        for p in problems:
            k = 4; n_ok = rng.randint(0, 4)
            samples = []
            for j in range(k):
                ok = j < n_ok
                text = fake_answer_text(p, ok, finish=True)
                g_ok, frac = grade(text, p)
                assert g_ok == ok, (p["id"], text[-60:], gold_list(p))
                clean = ok
                samples.append({"correct": g_ok, "frac": frac, "clean": clean, "n_tokens": len(text) + 100 * j,
                                "finish_reason": "stop", "completion_ids": tok(text)["input_ids"] if clean else None,
                                "text": text})
            f.write(json.dumps({"id": p["id"], "src": p["src"], "qtype": qtype(p), "k": k,
                                "n_correct": sum(s["correct"] for s in samples), "samples": samples}) + "\n")

    # ---- build SFT set (real script)
    sft_path = f"{tmp}/sft.jsonl"
    sys.argv = ["x", "--samples", samples_path, "--problems", problems_path, "--data", DATA, "--out", sft_path,
                "--openr1_n", "15", "--per_problem", "2", "--indomain_repeat", "2"]
    build_sft.main()
    sft = load_jsonl(sft_path)
    stats = json.load(open(sft_path.replace(".jsonl", "_stats.json")))
    assert stats["openr1_examples"] > 0 and stats["indomain_examples"] > 0
    for d in sft:
        assert d["prompt"].endswith("<|im_start|>assistant\n<think>\n")
        sid = d["id"]
        if not str(sid).startswith(("openr1_", "mmlupro_")):
            assert split[sid] == "train", f"leak: {sid} is {split[sid]}"
    r1 = [d for d in sft if d["source"] == "openr1"]
    r1_text = [tok.decode(d["completion_ids"]) for d in r1]
    assert all(not t.startswith("<think>") and not t.startswith("\n") and "\\boxed{42}" in t for t in r1_text)
    assert all("xxxxx" not in t for t in r1_text), "should pick the SHORTEST verified trace"
    print(f"[test] SFT rows={len(sft)} stats={stats}")

    # ---- SFT tokenization + masking (real function)
    rows, dropped = make_rows(tok, sft[:50], max_seq=100000)
    for r, d in zip(rows, sft[:50]):
        n_p = len(tok(d["prompt"])["input_ids"])
        assert r["labels"][:n_p] == [-100] * n_p and r["labels"][n_p:] == d["completion_ids"] + [151645]
        assert r["input_ids"][n_p:] == r["labels"][n_p:]
    rows2, dropped2 = make_rows(tok, sft[:50], max_seq=10)
    assert len(rows2) == 0 and dropped2 == 50
    print("[test] SFT loss mask: prompt masked, completion + <|im_end|> trained: OK")

    # ---- GRPO prompt set (real script)
    gp = f"{tmp}/grpo.jsonl"
    sys.argv = ["x", "--samples", samples_path, "--problems", problems_path, "--out", gp, "--max_prompt", "100000", "--mmlu_share", os.environ.get("MMLU_SHARE", "0.35")]
    build_grpo.main()
    g = load_jsonl(gp)
    by_row = {r["id"]: r for r in load_jsonl(samples_path)}
    for r in g:
        it = json.loads(r["item"]); base_id = str(it["id"]).replace("_shuf", "")
        orig = next(p for p in problems if str(p["id"]) == base_id)
        if not str(base_id).startswith("mmlupro_"):
            assert split[orig["id"]] == "train"
        nc = by_row[orig["id"]]["n_correct"]
        assert 0 < nc < 4 or nc == 4, nc
        # a correct, finished answer to this exact item must get reward 1
        assert reward_for(fake_answer_text(it, True), it, truncated=False) == 1.0
        assert reward_for(fake_answer_text(it, False), it, truncated=False) <= 0.5
    print(f"[test] GRPO prompts={len(g)}; every prompt rewards a correct answer with 1.0: OK")
    n_mm = sum(json.loads(r["item"])["id"].__str__().startswith("mmlupro_") for r in g)
    assert n_mm <= 0.35 / 0.65 * (len(g) - n_mm) + 1, (n_mm, len(g))
    print(f"[test] GRPO MMLU-Pro share {n_mm}/{len(g)} within 35% cap: OK")

    # ---- gen_samples honours MMLU_N on resume (smaller = stop early, larger = extend), never re-fetches needlessly
    import contextlib, io
    items_path, out_path = f"{tmp}/gs_items.jsonl", f"{tmp}/gs_out.jsonl"
    def gs(n):
        sys.argv = ["x", "--data", DATA, "--out", out_path, "--items_out", items_path, "--mmlu_n", str(n)]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            gen_samples.main()
        return buf.getvalue()
    n_train = sum(split[it["id"]] == "train" for it in items)
    open(items_path, "w").close()
    with open(out_path, "w") as f:   # pretend everything is already sampled so no GPU is needed
        for p in [dict(it, src="train") for it in items if split[it["id"]] == "train"] + gen_samples.mmlu_items(30, items):
            f.write(json.dumps({"id": p["id"], "samples": []}) + "\n")
    assert f"problems={n_train + 10} " in gs(10)
    assert f"problems={n_train + 5} " in gs(5) and len(load_jsonl(items_path)) == n_train + 10, "shrink must keep the file"
    assert f"problems={n_train + 20} " in gs(20) and len(load_jsonl(items_path)) == n_train + 20
    first10 = [p["id"] for p in load_jsonl(items_path) if p["src"] == "mmlupro"][:10]
    assert first10 == [m["id"] for m in gen_samples.mmlu_items(10, items)], "extension must keep the same first items"
    print("[test] gen_samples MMLU_N resume: shrink stops early, grow extends the same list: OK")

    # ---- reward wrapper as TRL calls it
    stats = {"n": 0, "correct": 0, "no_answer": 0, "cut_off": 0}
    fn = make_reward_fn(max_completion=50, stats=stats, cutoff="penalize")
    it = json.loads(g[0]["item"])
    comps = [fake_answer_text(it, True), fake_answer_text(it, True), fake_answer_text(it, False, finish=False)]
    out = fn(prompts=["p"] * 3, completions=comps, completion_ids=[[1] * 10, [1] * 50, [1] * 50],
             item=[g[0]["item"]] * 3, qtype=[g[0]["qtype"]] * 3, trainer_state=None)
    assert out[0] == 1.0 and out[1] == -0.5 and out[2] == -0.5, out
    conv = fn(prompts=["p"], completions=[[{"role": "assistant", "content": comps[0]}]], completion_ids=[[1]],
              item=[g[0]["item"]])
    assert conv == [1.0]
    fn_i = make_reward_fn(max_completion=50, stats=stats, cutoff="ignore")
    out_i = fn_i(prompts=["p"] * 3, completions=comps, completion_ids=[[1] * 10, [1] * 50, [1] * 49],
                 item=[g[0]["item"]] * 3)
    assert out_i == [1.0, 0.0, -0.5], out_i     # cut off -> neutral (masked by trl); finished w/o box -> -0.5
    print("[test] cutoff=ignore: cut-off rollout neutral, finished-without-answer still -0.5: OK")
    print("[test] reward wrapper: correct=1, hit token limit=-0.5, unfinished=-0.5, chat-format input OK")
    # ---- near-copy detection (reworded question, or same options in another order)
    from mathrl.common import NearDupIndex
    ndx = NearDupIndex(items)
    long_q = max(items, key=lambda it: len(it["question"]))
    words = long_q["question"].split()
    reworded = " ".join(words[:len(words) // 2] + ["indeed"] + words[len(words) // 2:]).upper() + " ??"
    assert ndx.is_dup(reworded), "reworded copy must be caught"
    mcq = next(it for it in items if it.get("options") and sum(len(str(o)) >= 4 for o in it["options"]) >= 6)
    assert ndx.is_dup("A totally different stem about nothing in particular here", list(reversed(mcq["options"])))
    others = [it["question"] for it in items[:50]]
    assert not ndx.is_dup("Compute the integral of a completely unrelated quantity over the unit square please")
    fp = sum(ndx.is_dup(f"mmlu q {i} about kinetic energy of a falling body with mass {i} kilograms") for i in range(50))
    assert fp == 0, fp
    print("[test] near-copy filter: reworded + reordered-options copies caught, unrelated text passes: OK")

    # ---- RL prompt re-check with the SFT model: keep mixed, keep long all-right, drop all-right / all-wrong
    from mathrl.recheck_grpo import select
    mk = lambda i: {"prompt": "p", "item": json.dumps({"id": i}), "qtype": "free_single"}
    rows_ = [mk(i) for i in range(6)]
    checks = {0: {"n_correct": 2, "min_correct_tokens": 100}, 1: {"n_correct": 4, "min_correct_tokens": 100},
              2: {"n_correct": 0, "min_correct_tokens": None}, 3: {"n_correct": 4, "min_correct_tokens": 15000},
              4: {"n_correct": 1, "min_correct_tokens": 9000}}
    keep, st = select(rows_, checks, 4, True, 16384, min_keep=1)
    assert [json.loads(r["item"])["id"] for r in keep] == [0, 3, 4, 5], keep
    assert st == {"unchecked": 1, "mixed": 2, "long": 1, "all_right": 1, "all_wrong": 1}, st
    keep2, _ = select(rows_, checks, 4, True, 16384, min_keep=10)
    assert keep2 == rows_, "too few left -> fall back to the original set"
    print("[test] RL re-check: keeps mixed + long, drops all-right/all-wrong, falls back if too few: OK")

    # ---- compute-unit guard persists across restarts; paired comparison basics
    from mathrl.common import TimeBudget, paired_diff
    tp = f"{tmp}/time.json"
    b = TimeBudget(tp, 2 / 3600); b.used0 = 1.5; b.save()
    b2 = TimeBudget(tp, 2 / 3600)
    assert 1.5 <= b2.used() < 1.6 and b2.would_exceed(1.0) and not b2.would_exceed(0.2)
    assert not TimeBudget(tp + "x", 0).would_exceed(1e9)
    c = paired_diff({1: True, 2: False, 3: False}, {1: True, 2: True, 3: False, 4: True})
    assert c["n"] == 3 and abs(c["diff"] - 100 / 3) < 1e-9 and c["fixed"] == 1 and c["broke"] == 0
    print("[test] time limit survives restarts; paired diff counts fixed/broke on shared questions: OK")
    print("ALL PIPELINE TESTS PASS")


if __name__ == "__main__":
    main()
