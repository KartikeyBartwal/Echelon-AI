from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Use a smaller text-only model
model_name = "deepseek-ai/deepseek-coder-1.3b-base"  # This is much smaller than the 7B version

# Enable model optimization
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use half precision
    low_cpu_mem_usage=True,     # Optimize CPU memory usage
    device_map="auto"           # Automatically handle device placement
)

# Test the model
prompt = "Greet me in the style of a pirate"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_length=512,
    num_return_sequences=1,
    temperature=0.7,
    do_sample=True
)
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded_output)