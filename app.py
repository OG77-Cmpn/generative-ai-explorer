import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_text(prompt: str, max_new_tokens: int = 50) -> str:
    """Generates text based on a user prompt using the open-source DistilGPT2 model."""
    model_name = "distilgpt2"

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    print(f"Generating completion for prompt: '{prompt}'\n")
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    user_prompt = "Artificial Intelligence will transform the future of"
    result = generate_text(user_prompt)

    print("--- Output ---")
    print(result)