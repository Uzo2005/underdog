from transformers import AutoModelForCausalLM, AutoTokenizer

# Provide the local paths to the model and tokenizer
model_path = "./stablelm_1_6b_model/"
tokenizer_path = "./stablelm_1_6b_model/"

# Load tokenizer from local path
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Load model from local path
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
)
# model.cuda()

# Continue with the rest of your code as before
inputs = tokenizer("The weather is always wonderful", return_tensors="pt").to(model.device)
tokens = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.70,
    top_p=0.95,
    do_sample=True,
)
print(tokenizer.decode(tokens[0], skip_special_tokens=True))
