"""Merge the SFT LoRA into the base weights so GRPO starts from the SFT model (plain transformers + peft)."""
import argparse, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(os.path.join(a.out, "config.json")):
        print(f"[merge] already merged at {a.out}"); return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from mathrl.common import BASE_MODEL
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, a.adapter).merge_and_unload()
    model.save_pretrained(a.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(BASE_MODEL).save_pretrained(a.out)
    print(f"[merge] DONE -> {a.out}")


if __name__ == "__main__":
    main()
