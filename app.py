import torch
import streamlit as st
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    BlipForQuestionAnswering,
)
from PIL import Image

### Load the model and tokenizer for extracting the description of an image
model_description = AutoModelForCausalLM.from_pretrained(
    "MILVLG/imp-v1-3b",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer_description = AutoTokenizer.from_pretrained(
    "MILVLG/imp-v1-3b", trust_remote_code=True
)

### Context for the description
text = """A chat between a curious user and an artificial intelligence assistant.
The assistant gives useful, detailed and polite answers to the user's questions.
The assistant is an expert in the field of analyzing images that people are going to post on their social networks.
The assistant must respond taking into account the following issues:
Emotional context of the image:
Professional context of the image:
USER: <image>\nTell me about the emotional context of this image and the professional context of the image.
ASSISTANT:"""
image = Image.open("meu path")

### Generate the description of the image
input_ids_description = tokenizer_description(
    text, return_tensors="pt"
).input_ids_description
image_tensor = model_description.image_preprocess(image)
output_ids_description = model_description.generate(
    input_ids_description, max_new_tokens=300, images=image_tensor, use_cache=True
)[0]

descriction = tokenizer_description.decode(
    output_ids_description[input_ids_description.shape[1] :], skip_special_tokens=True
).strip()

### Conversational Chain

model_QA = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
processor_QA = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
inputs_QA = processor_QA(images=image, text=text, return_tensors="pt")
outputs_QA = model_QA.generate(**inputs_QA)
