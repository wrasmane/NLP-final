import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

SKIT_PATH = "./models/skit_model"
ONE_LINER_PATH = "./models/one_liner_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

skit_model = AutoModelForCausalLM.from_pretrained(SKIT_PATH, torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32).to(DEVICE)
skit_tokenizer = AutoTokenizer.from_pretrained(SKIT_PATH)
skit_pipeline = pipeline("text-generation", model=skit_model, tokenizer=skit_tokenizer, device=0 if DEVICE == "cuda" else -1)
def generate_skit(setup):
    prompt = "Write a stand-up comedy skit about " + setup + ": <|startofjoke|>" if setup else "Write a stand-up comedy skit: <|startofjoke|>"
    results = skit_pipeline(
        prompt,
        max_new_tokens=120,
        do_sample=True,
        top_k=40,
        top_p=0.92,
        temperature=0.7,
        repetition_penalty=1.3,
        num_return_sequences=1,
        pad_token_id=skit_tokenizer.eos_token_id
    )
    output_text = results[0]['generated_text']
    if "<|startofjoke|>" in output_text:
        joke_section = output_text.split("<|startofjoke|>", 1)[-1]
    else:
        joke_section = output_text

    if "<|endofjoke|>" in joke_section:
        joke = joke_section.split("<|endofjoke|>", 1)[0].strip()
    else:
        joke = joke_section.strip()
    return joke

one_liner_model = AutoModelForCausalLM.from_pretrained(ONE_LINER_PATH).to(DEVICE)
one_liner_tokenizer = AutoTokenizer.from_pretrained(ONE_LINER_PATH)
def generate_one_liner(setup):
    prompt = "Generate one joke from the following theme: Best " + setup + " Jokes"
    one_liner_model.eval()
    input_ids = one_liner_tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = one_liner_model.generate(
            input_ids,
            max_length=200,
            temperature=0.9,
            top_p=0.95,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=one_liner_tokenizer.eos_token_id
        )
    generated_text = one_liner_tokenizer.decode(output[0], skip_special_tokens=True)
    generated_joke = generated_text.replace(prompt, "").strip()
    return generated_joke


def main():
    print("\n\nWelcome to the AI Joke Generator Prototype\n")
    joke_type = input("What type of joke would you like to generate? (skit(1) or one_liner(2)): ")
    setup = input("Provide the topic to the joke: ")
    print("")
    cont = "y"
    while cont == "y":
        if joke_type == "skit" or joke_type == "1":
            print(generate_skit(setup))
        elif joke_type == "one_liner" or joke_type == "2":
            print(generate_one_liner(setup))
        cont = input("Continue? (y/n): ")
        print("")

if __name__ == "__main__":
    main()