from transformers import AutoTokenizer, AutoModelForCausalLM

# Local model path
MODEL_PATH = "./models/qwen"

print("Loading model...")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH
)

print("Model loaded successfully!")
print("-" * 50)

# Ask a question
prompt = "Explain Java microservices in simple terms."

# Convert prompt to tokens
inputs = tokenizer(prompt, return_tensors="pt")

# Generate response
outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.7,
    do_sample=True
)

# Convert tokens back to text
response = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print("\nQuestion:")
print(prompt)

print("\nModel Response:")
print(response)