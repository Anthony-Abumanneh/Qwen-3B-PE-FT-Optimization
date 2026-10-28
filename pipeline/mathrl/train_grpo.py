"""GRPO on top of the SFT model. Reward = the same grader used for evaluation.

Design (see notebook for reasoning):
  * reward: +1 correct, partial credit on multi-blank, 0 wrong, -0.5 finished without a final \\boxed{} answer
  * loss_type=dr_grpo (no length bias), beta=0 (no KL / no reference model), clip-higher eps_high=0.28
  * rollouts cut off at the RL budget (8k) are, by default, left out of the loss (DAPO "overlong filtering"):
    eval allows 16k, so RL must not teach the model to cut short reasoning that would have finished by 16k.
    --cutoff penalize instead gives them -0.5 (pushes for shorter answers).
  * prompts pre-filtered so each group has a real mix of right and wrong answers
Checkpoints every --save_steps to --ckpt_dir (Drive); rerun resumes.
"""
import argparse, glob, json, os, shutil

os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")


def make_reward_fn(max_completion, stats, cutoff="ignore"):
    from mathrl.common import reward_for, R_WRONG

    def correctness_reward(prompts, completions, completion_ids=None, item=None, **kw):
        out = []
        for i, c in enumerate(completions):
            text = c if isinstance(c, str) else c[-1]["content"]
            truncated = completion_ids is not None and len(completion_ids[i]) >= max_completion
            if truncated and cutoff == "ignore":
                r = R_WRONG          # neutral for the group baseline; trl drops it from the loss (mask_truncated_completions)
            else:
                r = reward_for(text, json.loads(item[i]), truncated)
            out.append(r)
            stats["n"] += 1; stats["correct"] += r == 1.0; stats["no_answer"] += r < 0; stats["cut_off"] += truncated
        return out
    return correctness_reward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="merged SFT model dir (or base model id to skip SFT)")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--adapter_out", required=True)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--prompts_per_step", type=int, default=2)
    ap.add_argument("--max_prompt", type=int, default=2048)
    ap.add_argument("--max_completion", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--save_steps", type=int, default=10)
    ap.add_argument("--keep_steps", default="20,40", help="comma list of steps whose adapter is kept for dev eval")
    ap.add_argument("--gpu_util", type=float, default=0.85)
    ap.add_argument("--cutoff", choices=["ignore", "penalize"], default="ignore",
                    help="rollouts that hit --max_completion: leave out of the loss (default) or reward -0.5")
    ap.add_argument("--max_hours", type=float, default=0, help="stop cleanly before this much training time (total over resumes); 0 = no limit")
    a = ap.parse_args()

    from unsloth import FastLanguageModel           # before trl/transformers
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer
    import time
    from mathrl.common import LORA_RANK, load_jsonl, TimeBudget

    model, tok = FastLanguageModel.from_pretrained(
        model_name=a.base, max_seq_length=a.max_prompt + a.max_completion, load_in_4bit=False,
        load_in_8bit=False, dtype=torch.bfloat16, fast_inference=True, max_lora_rank=LORA_RANK,
        gpu_memory_utilization=a.gpu_util)
    model = FastLanguageModel.get_peft_model(
        model, r=LORA_RANK, lora_alpha=2 * LORA_RANK, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=3407)

    rows = load_jsonl(a.prompts)
    need = a.steps * a.prompts_per_step
    print(f"[grpo] {len(rows)} prompts available, {need} needed for {a.steps} steps"
          + ("" if len(rows) >= need else " -> some prompts will repeat (more than 1 epoch)"))
    ds = Dataset.from_list(rows)

    stats = {"n": 0, "correct": 0, "no_answer": 0, "cut_off": 0}
    correctness_reward = make_reward_fn(a.max_completion, stats, a.cutoff)

    keep = {int(s) for s in a.keep_steps.split(",") if s.strip()}
    os.makedirs(a.ckpt_dir, exist_ok=True)
    budget = TimeBudget(os.path.join(a.ckpt_dir, "time_used.json"), a.max_hours)
    clock = {"last": time.time(), "times": []}
    hist, zstd = [], []     # per-step (acc, no_answer, cut_off) and trl's share of groups with no reward spread

    def health(step, per_step):
        """Plain-language check every 10 steps; only warns, never stops the run."""
        h = hist[-10:]
        acc, na, co = (sum(x[i] for x in h) / len(h) for i in range(3))
        zs = sum(zstd[-10:]) / len(zstd[-10:]) if zstd else None
        msg = f"[grpo] HEALTH @ step {step}: last {len(h)} steps acc={acc:.2f} no_answer={na:.2f} cut_off={co:.2f}"
        if zs is not None:
            msg += f" groups_without_signal={zs:.2f}"
        print(msg, flush=True)
        if len(hist) >= 20:
            first = sum(x[0] for x in hist[:10]) / 10
            print(f"[grpo]   accuracy: first 10 steps {first:.2f} -> last 10 steps {acc:.2f} (about +/-0.04 of noise)", flush=True)
        warn = []
        if zs is not None and zs > 0.6:
            warn.append("most groups have no reward spread, so little is being learned (prompts too easy or too hard)")
        if co > 0.35:
            warn.append("over a third of answers are cut off at the token limit")
        if len(hist) >= 20 and acc < sum(x[0] for x in hist[:10]) / 10 - 0.12:
            warn.append("training accuracy fell clearly; consider stopping and keeping the SFT model")
        if per_step / 60 > 14:
            warn.append(f"{per_step / 60:.0f} min/step means fewer than {int(a.max_hours * 60 / (per_step / 60)) if a.max_hours else 0} steps fit in the limit")
        for w in warn:
            print("[grpo]   WARNING: " + w, flush=True)

    class Keep(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "frac_reward_zero_std" in logs:
                zstd.append(float(logs["frac_reward_zero_std"]))

        def on_train_begin(self, args, state, control, **kw):
            clock["last"] = time.time()   # step timing excludes model loading / vLLM warm-up

        def on_step_end(self, args, state, control, **kw):
            if state.global_step in keep:
                d = os.path.join(os.path.dirname(a.adapter_out.rstrip("/")), f"grpo_adapter_step{state.global_step}")
                model.save_pretrained(d); tok.save_pretrained(d)
                print(f"[grpo] kept adapter for dev eval: {d}", flush=True)
            if stats["n"]:
                print(f"[grpo] step {state.global_step}: acc={stats['correct']/stats['n']:.2f} "
                      f"no_answer={stats['no_answer']/stats['n']:.2f} cut_off={stats['cut_off']/stats['n']:.2f} "
                      f"over {stats['n']} samples", flush=True)
                hist.append((stats["correct"] / stats["n"], stats["no_answer"] / stats["n"], stats["cut_off"] / stats["n"]))
                stats.update(n=0, correct=0, no_answer=0, cut_off=0)
            now = time.time()
            clock["times"].append(now - clock["last"]); clock["last"] = now
            budget.save()
            recent = clock["times"][-5:]
            per_step = sum(recent) / len(recent)
            left = state.max_steps - state.global_step
            if state.global_step % 5 == 0 or len(clock["times"]) == 3:
                print(f"[grpo] {per_step / 60:.1f} min/step -> ~{left * per_step / 3600:.1f} h for the remaining {left} steps "
                      f"({budget.used() / 3600:.1f} h used" + (f" of {a.max_hours} h limit)" if a.max_hours else ")"), flush=True)
            if hist and state.global_step % 10 == 0:
                health(state.global_step, per_step)
            if left > 0 and budget.would_exceed(per_step):
                print(f"[grpo] STOPPING at step {state.global_step}: the next step would pass the {a.max_hours} h limit. "
                      "The adapter is saved as the final one.", flush=True)
                control.should_training_stop = True
            return control

    cfg = GRPOConfig(
        output_dir=a.ckpt_dir, max_steps=a.steps, learning_rate=a.lr, lr_scheduler_type="constant_with_warmup",
        # constant LR (as in DAPO / Dr. GRPO): stopping early at the unit limit leaves no half-finished schedule
        warmup_steps=5, weight_decay=0.0, max_grad_norm=1.0, optim="adamw_torch", adam_beta2=0.99,
        bf16=True, logging_steps=1, save_strategy="steps", save_steps=a.save_steps, save_total_limit=2,
        report_to="none", seed=3407,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=a.num_generations * a.prompts_per_step,
        num_generations=a.num_generations,
        max_prompt_length=a.max_prompt, max_completion_length=a.max_completion,
        temperature=a.temperature, top_p=0.95, top_k=None, min_p=0.0,
        loss_type="dr_grpo", beta=0.0, epsilon=0.2, epsilon_high=0.28,
        scale_rewards=False, mask_truncated_completions=(a.cutoff == "ignore"),
    )
    trainer = GRPOTrainer(model=model, processing_class=tok, reward_funcs=[correctness_reward],
                          args=cfg, train_dataset=ds, callbacks=[Keep()])
    ckpts = sorted(glob.glob(os.path.join(a.ckpt_dir, "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[1]))
    resume = ckpts[-1] if ckpts else None
    print(f"[grpo] resume_from={resume}")
    trainer.train(resume_from_checkpoint=resume)
    model.save_pretrained(a.adapter_out)
    tok.save_pretrained(a.adapter_out)
    with open(os.path.join(a.adapter_out, "train_log.json"), "w") as f:
        json.dump(trainer.state.log_history, f)
    print(f"[grpo] DONE adapter saved to {a.adapter_out}")


if __name__ == "__main__":
    main()
