import argparse
import json
import os
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file
from configuration_lora_moe import LoraMoeConfig
from modelling import LoraMoeModel
import time 
BASE_MODEL_ID  = "Qwen/Qwen2.5-Coder-3B-Instruct"
CHECKPOINT     = "/teamspace/studios/this_studio/outputs/malora/checkpoint-383"
OUTPUT_DIR     = "/teamspace/studios/this_studio/evalplus_outputs"
MAX_NEW_TOKENS = 512


def load_model():
    print(f"Loading {BASE_MODEL_ID}...")
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

    base_config = AutoConfig.from_pretrained(BASE_MODEL_ID)
    config = LoraMoeConfig(
        experts_rank         = 8,
        attention_rank       = 32,
        num_local_experts    = 8,
        num_experts_per_tok  = 2,
        router_aux_loss_coef = 0.001,
        use_attention_lora   = False,
        **base_config.to_dict(),
    )

    wrapped = LoraMoeModel(base, config)
    ckpt_sd = load_file(f"{CHECKPOINT}/model.safetensors")
    wrapped.load_state_dict(ckpt_sd, strict=True)
    wrapped = wrapped.to("cuda")
    wrapped.eval()
    print("Done.\n")
    return wrapped, tokenizer


def format_humaneval_prompt(problem, tokenizer):
    messages = [
        {"role": "system", "content": "You are a Python coding assistant. Complete the function below. Return ONLY the function body — no explanation, no markdown, no extra text."},
        {"role": "user",   "content": f"Complete this Python function:\n\n{problem['prompt']}"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def format_mbpp_prompt(problem, tokenizer):
    tests = "\n".join(problem.get("test_list", [])[:2])
    messages = [
        {"role": "system", "content": "You are a Python coding assistant. Write a Python function that solves the given task. Return ONLY the function — no explanation, no markdown, no extra text."},
        {"role": "user",   "content": f"Task: {problem['prompt']}\n\nExample tests:\n{tests}"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def strip_markdown_fences(code):
    lines = code.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


@torch.inference_mode()
def generate_completion(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]
    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)


def run_humaneval(model, tokenizer):
    from evalplus.data import get_human_eval_plus
    problems = get_human_eval_plus()
    print(f"HumanEval+: {len(problems)} problems")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []
    for i, (task_id, problem) in enumerate(problems.items()):
        start = time.time()
        completion = strip_markdown_fences(generate_completion(model, tokenizer, format_humaneval_prompt(problem, tokenizer)))
        print(f"  problem {i+1} done in {time.time()-start:.1f}s")
        results.append({"task_id": task_id, "completion": completion})
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(problems)}]")

    out_path = os.path.join(OUTPUT_DIR, "humaneval_completions.jsonl")
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved → {out_path}\n")
    return out_path


def run_mbpp(model, tokenizer):
    from evalplus.data import get_mbpp_plus
    problems = get_mbpp_plus()
    print(f"MBPP+: {len(problems)} problems")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []
    for i, (task_id, problem) in enumerate(problems.items()):
        completion = strip_markdown_fences(generate_completion(model, tokenizer, format_mbpp_prompt(problem, tokenizer)))
        results.append({"task_id": task_id, "completion": completion})
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(problems)}]")

    out_path = os.path.join(OUTPUT_DIR, "mbpp_completions.jsonl")
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved → {out_path}\n")
    return out_path


def score(dataset, completions_path):
    print(f"Scoring {dataset}...")
    os.system(f"python -m evalplus.evaluate --dataset {dataset} --samples {completions_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["humaneval", "mbpp", "both"], default="both")
    args = parser.parse_args()

    model, tokenizer = load_model()

    if args.dataset in ("humaneval", "both"):
        score("humaneval", run_humaneval(model, tokenizer))

    if args.dataset in ("mbpp", "both"):
        score("mbpp", run_mbpp(model, tokenizer))
