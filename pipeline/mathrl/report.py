"""One results table: every run re-graded with the same grader on the same questions."""
import argparse, glob, json, os
from mathrl.common import load_jsonl, make_split, grade, qtype, summarize, paired_diff


def sample_files(path):
    """The main results file plus any extra-sample files written next to it (pe_test_s1.jsonl, ...)."""
    return [path] + sorted(glob.glob(path.replace(".jsonl", "_s*.jsonl")))


def score_file(path, items_by_id, split, which, label):
    n_expected = sum(1 for v in split.values() if v == which)
    files = sample_files(path)
    all_rows, per_q, complete = [], {}, True
    for fp in files:
        rows = {}
        for r in load_jsonl(fp):
            it = items_by_id.get(r["id"])
            if it is None or split[it["id"]] != which:
                continue
            ok, frac = grade(r["response"], it)
            rows[r["id"]] = {"id": r["id"], "qtype": qtype(it), "correct": ok, "frac": frac,
                             "finish_reason": r.get("finish_reason", "length" if "</think>" not in r["response"] else "stop")}
        complete &= len(rows) == n_expected
        all_rows += rows.values()
        for i, r in rows.items():
            per_q.setdefault(i, []).append(float(r["correct"]))
    s = summarize(all_rows, label)            # with k samples: average accuracy over all k x n answers
    s["n"] = len(per_q)
    s["k"] = len(files)
    s["complete"] = complete and len(per_q) == n_expected
    s["per_q"] = {i: sum(v) / len(v) for i, v in per_q.items()}
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="label=path.jsonl ...")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    items = load_jsonl(a.data)
    by_id = {it["id"]: it for it in items}
    split = make_split(items)
    print(f"\n{'run':<34}{'n':>5}{'MCQ':>8}{'single':>8}{'multi':>8}{'OVERALL':>9}{'trunc%':>8}")
    print("-" * 80)
    scored = []
    for spec in a.runs:
        label, path = spec.split("=", 1)
        if not os.path.exists(path):
            print(f"{label:<34}  (not run yet)"); continue
        s = score_file(path, by_id, split, a.split, label)
        f = lambda v: f"{v:7.1f}%" if v is not None else "    n/a"
        flag = ("" if s["complete"] else "  <- incomplete") + (f"  (avg of {s['k']} samples)" if s["k"] > 1 else "")
        print(f"{label:<34}{s['n']:>5}{f(s['mcq'])}{f(s['free_single'])}{f(s['free_multi'])}{s['overall']:8.1f}%{s['truncated_pct']:7.1f}%{flag}")
        scored.append((label, s))
    # Paired comparisons on the same questions: is a change bigger than sampling luck?
    pairs = [(scored[i - 1], scored[i]) for i in range(1, len(scored))]
    labels = [l for l, _ in scored]
    if "PE" in labels:                                  # also PE -> each later stage (e.g. PE -> PE+SFT+RL)
        i = labels.index("PE")
        pairs += [(scored[i], x) for x in scored[i + 2:]]
    if pairs:
        print(f"\npaired change on the same questions (95% CI; 'fixed'/'broke' = questions that flipped):")
        for (la, sa), (lb, sb) in pairs:
            c = paired_diff(sa["per_q"], sb["per_q"])
            if c:
                verdict = "clear gain" if c["lo"] > 0 else ("clear loss" if c["hi"] < 0 else "within noise")
                print(f"  {la} -> {lb}: {c['diff']:+.1f} pts  [{c['lo']:+.1f}, {c['hi']:+.1f}]  "
                      f"fixed {c['fixed']}, broke {c['broke']}  -> {verdict}")
    print(f"\nsplit = {a.split}; grader = mathrl/common.py grade(); baseline used 8,192 max tokens, all others 16,384.")


if __name__ == "__main__":
    main()
