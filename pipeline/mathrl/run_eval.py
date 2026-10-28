"""Evaluate a model (optionally + LoRA adapter) on a split with the frozen protocol.

python -m mathrl.run_eval --split test --out RESULTS/pe_test.jsonl
python -m mathrl.run_eval --split test --lora ADAPTER_DIR --out RESULTS/sft_test.jsonl
Resumable: rerunning skips questions already in --out.
"""
import argparse, json, os, time
from mathrl.common import (BASE_MODEL, EVAL_MAX_TOKENS, EVAL_SAMPLING, LORA_RANK, load_jsonl, make_split,
                           prompt_text, qtype, gold_list, grade, done_keys, append_jsonl, summarize)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test", choices=["test", "dev", "train", "all"])
    ap.add_argument("--base", default=BASE_MODEL, help="HF id or local merged-model dir")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--max_tokens", type=int, default=EVAL_MAX_TOKENS)
    ap.add_argument("--chunk", type=int, default=128, help="questions per saved chunk")
    ap.add_argument("--gpu_util", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=0, help="sampling seed; extra test samples use 1, 2, 3 ...")
    ap.add_argument("--prompts", choices=["pe", "starter"], default="pe", help="system prompts: PE (default) or the original baseline's")
    a = ap.parse_args()

    items = load_jsonl(a.data)
    split = make_split(items)
    sel = [it for it in items if a.split == "all" or split[it["id"]] == a.split]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    done = done_keys(a.out)
    todo = [it for it in sel if it["id"] not in done]
    print(f"[eval] split={a.split} n={len(sel)} done={len(done & {i['id'] for i in sel})} todo={len(todo)} lora={a.lora}")

    if todo:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        tok = AutoTokenizer.from_pretrained(BASE_MODEL)          # one tokenizer/template for every model
        prompts = {it["id"]: prompt_text(tok, it, a.prompts) for it in todo}
        max_prompt = max(len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts.values())
        max_len = max_prompt + a.max_tokens + 64
        print(f"[eval] longest prompt {max_prompt} tok -> max_model_len {max_len}")
        kw = dict(model=a.base, tokenizer=BASE_MODEL, max_model_len=max_len, gpu_memory_utilization=a.gpu_util,
                  seed=a.seed, enable_prefix_caching=True)
        lora_req = None
        if a.lora:
            from vllm.lora.request import LoRARequest
            kw.update(enable_lora=True, max_lora_rank=max(LORA_RANK, 64), max_loras=1)
            lora_req = LoRARequest("adapter", 1, os.path.abspath(a.lora))
        llm = LLM(**kw)
        sp = SamplingParams(n=1, max_tokens=a.max_tokens, **EVAL_SAMPLING)
        t0 = time.time()
        for s in range(0, len(todo), a.chunk):
            chunk = todo[s:s + a.chunk]
            outs = llm.generate([prompts[it["id"]] for it in chunk], sp, lora_request=lora_req)
            rows = []
            for it, o in zip(chunk, outs):
                c = o.outputs[0]
                ok, frac = grade(c.text, it)
                rows.append({"id": it["id"], "split": split[it["id"]], "qtype": qtype(it), "gold": gold_list(it),
                             "response": c.text, "n_tokens": len(c.token_ids), "finish_reason": c.finish_reason,
                             "correct": ok, "frac": frac})
            append_jsonl(a.out, rows)
            print(f"[eval] saved {s + len(chunk)}/{len(todo)}  ({(time.time() - t0) / 60:.1f} min)", flush=True)

    # Always re-grade from the file so every model is scored by the same, current grader.
    by_id = {it["id"]: it for it in sel}
    rows = []
    for r in load_jsonl(a.out):
        if r["id"] in by_id:
            r["correct"], r["frac"] = grade(r["response"], by_id[r["id"]])
            rows.append(r)
    rows = list({r["id"]: r for r in rows}.values())
    s = summarize(rows, a.label or os.path.basename(a.out))
    with open(a.out.replace(".jsonl", "_summary.json"), "w") as f:
        json.dump(s, f, indent=1)
    print("[eval] RESULT", json.dumps(s))


if __name__ == "__main__":
    main()
