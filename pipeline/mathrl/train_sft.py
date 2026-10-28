"""LoRA SFT on Qwen3-4B-Thinking-2507 (the same model as the baseline), loss on completion tokens only.

Checkpoints go to --ckpt_dir (put it on Drive); rerunning resumes from the latest checkpoint.
"""
import argparse, glob, json, os, random


def make_rows(tok, data, max_seq):
    """prompt ids (masked) + completion ids + <|im_end|> (trained). Exactly what the model sees/produces."""
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    rows, dropped = [], 0
    for d in data:
        p_ids = tok(d["prompt"], add_special_tokens=False)["input_ids"]
        c_ids = list(d["completion_ids"]) + [eos]
        if len(p_ids) + len(c_ids) > max_seq:
            dropped += 1
            continue
        rows.append({"input_ids": p_ids + c_ids, "labels": [-100] * len(p_ids) + c_ids})
    return rows, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft_data", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--adapter_out", required=True)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--max_seq", type=int, default=16384)
    ap.add_argument("--save_steps", type=int, default=25)
    a = ap.parse_args()

    from unsloth import FastLanguageModel          # must be imported before transformers
    import torch
    from transformers import Trainer, TrainingArguments
    from mathrl.common import BASE_MODEL, LORA_RANK, load_jsonl

    model, tok = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=a.max_seq, load_in_4bit=False, load_in_8bit=False,
        dtype=torch.bfloat16, fast_inference=False)
    model = FastLanguageModel.get_peft_model(
        model, r=LORA_RANK, lora_alpha=2 * LORA_RANK, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=3407)

    eos = tok.convert_tokens_to_ids("<|im_end|>")
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.convert_tokens_to_ids("<|endoftext|>")
    rows, dropped = make_rows(tok, load_jsonl(a.sft_data), a.max_seq)
    assert rows, "empty SFT dataset"
    n_lab = sum(sum(1 for x in r["labels"] if x != -100) for r in rows)
    print(f"[sft] {len(rows)} examples ({dropped} dropped as too long), {n_lab/1e6:.2f}M trained tokens")
    r0 = rows[0]
    n_prompt = r0["labels"].index(next(x for x in r0["labels"] if x != -100))
    print("[sft] prompt/completion boundary ->", repr(tok.decode(r0["input_ids"][max(0, n_prompt - 12):n_prompt + 12])))

    class DS(torch.utils.data.Dataset):
        def __len__(self): return len(rows)
        def __getitem__(self, i): return rows[i]

    def collate(batch):
        L = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), L), pad, dtype=torch.long)
        lab = torch.full((len(batch), L), -100, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n] = torch.tensor(b["input_ids"]); lab[i, :n] = torch.tensor(b["labels"]); att[i, :n] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}

    # Memory check: one forward+backward on the longest example before training, so an out-of-memory
    # error shows up in the first minute instead of hours into the run. Gradients are discarded.
    longest = max(rows, key=lambda r: len(r["input_ids"]))
    batch = {k: v.to(model.device) for k, v in collate([longest]).items()}
    model.train()
    model(**batch).loss.backward()
    model.zero_grad(set_to_none=True)
    del batch
    torch.cuda.empty_cache()
    print(f"[sft] memory check passed on the longest example ({len(longest['input_ids'])} tokens), "
          f"peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)

    args = TrainingArguments(
        output_dir=a.ckpt_dir, per_device_train_batch_size=1, gradient_accumulation_steps=a.grad_accum,
        num_train_epochs=1, learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.05,
        weight_decay=0.0, max_grad_norm=1.0, optim="adamw_8bit", bf16=True, logging_steps=5,
        save_strategy="steps", save_steps=a.save_steps, save_total_limit=2, report_to="none", seed=3407,
        dataloader_num_workers=0, remove_unused_columns=False, gradient_checkpointing=False)
    trainer = Trainer(model=model, args=args, train_dataset=DS(), data_collator=collate)
    ckpts = sorted(glob.glob(os.path.join(a.ckpt_dir, "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[1]))
    resume = ckpts[-1] if ckpts else None
    print(f"[sft] resume_from={resume}")
    trainer.train(resume_from_checkpoint=resume)
    model.save_pretrained(a.adapter_out)
    tok.save_pretrained(a.adapter_out)
    with open(os.path.join(a.adapter_out, "train_log.json"), "w") as f:
        json.dump(trainer.state.log_history, f)
    print(f"[sft] DONE adapter saved to {a.adapter_out}")


if __name__ == "__main__":
    main()
