from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the BabyLLaMA model and tokenizer from Hugging Face
model_name = "babylm/babyllama-100m-2024"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.save_pretrained("./model/babyllama-100m-2024")
tokenizer.save_pretrained("./model/babyllama-100m-2024")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)