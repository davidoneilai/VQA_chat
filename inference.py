import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import argparse
import os

torch.set_default_device("cuda")

model = AutoModelForCausalLM.from_pretrained(
    "MILVLG/imp-v1-3b",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("MILVLG/imp-v1-3b", trust_remote_code=True)

# Definir argumentos para o script
parser = argparse.ArgumentParser(description="Ask a question about an image.")
parser.add_argument("query", help="The question to ask about the image.")
parser.add_argument("image_path", help="Path to the image file.")
args = parser.parse_args()

# Verificar se o arquivo da imagem existe
if not os.path.exists(args.image_path):
    raise FileNotFoundError(f"The image file {args.image_path} does not exist.")

text = f"""A chat between a curious user and an artificial intelligence assistant.
The assistant gives useful, detailed and polite answers to the user's questions.
The assistant is an expert in the field of analyzing images that people are going to post on their social networks.
The assistant must respond taking into account the following issues:
Emotional context of the image:
Professional context of the image:
USER: <image>\n{args.query}
ASSISTANT:"""
image = Image.open(args.image_path)

input_ids = tokenizer(text, return_tensors="pt").input_ids
image_tensor = model.image_preprocess(image)
output_ids = model.generate(
    input_ids, max_new_tokens=300, images=image_tensor, use_cache=True
)[0]
response = tokenizer.decode(
    output_ids[input_ids.shape[1] :], skip_special_tokens=True
).strip()
print(response)
