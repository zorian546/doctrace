"""
Local text generation with Qwen2.5-3B-Instruct.

Setup notes:
- Precision and memory: bfloat16 weights (~6GB, unquantised) fit in 8GB of VRAM next
  to BGE-M3 and the cross-encoder. Skipping quantisation also avoids extra kernel
  dependencies on Blackwell (sm_120) GPUs.
- Determinism: greedy decoding (`do_sample=False`), so repeated evaluation runs give
  identical output.
- Prefill: the reply can be forced to begin with given text (for example "[" to
  start a JSON array) for structured-output prompts.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
_model = None
_tokenizer = None


def _load():
    global _model, _tokenizer
    if _model is None:
        print(f"Loading {_MODEL_NAME} (first call only, may take a while)...")
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(
            _MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        print("Model loaded.")
    return _model, _tokenizer


def complete(system_prompt: str | None, user_message: str,
             max_new_tokens: int = 500, prefill: str | None = None) -> str:
    """Run one chat turn through the local model and return the reply text.

    system_prompt: optional system message, None to leave it out.
    user_message: the user turn.
    prefill: optional text the reply must start with; it is added back to the
    returned string.
    """
    model, tokenizer = _load()

    turns = []
    if system_prompt:
        turns.append({"role": "system", "content": system_prompt})
    turns.append({"role": "user", "content": user_message})

    prompt_text = tokenizer.apply_chat_template(
        turns, tokenize=False, add_generation_prompt=True
    )
    if prefill:
        prompt_text += prefill

    encoded = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = output_ids[0][encoded["input_ids"].shape[1]:]
    reply = tokenizer.decode(new_ids, skip_special_tokens=True)

    return (prefill + reply) if prefill else reply
