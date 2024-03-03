import os
from fastapi import FastAPI, UploadFile, File, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from inference import model, tokenizer, input_ids, image_tensor, output_ids
from uuid import uuid4
import shutil
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if not os.path.exists("tmp"):
    os.mkdir("tmp")

images = {}


@app.post("/image/ingest")
async def store_file(file: UploadFile = File(...)):

    global index

    try:

        print(file.filename)
        id = str(uuid4())
        file_location = f"tmp/{id}"

        if not os.path.exists(file_location):
            os.mkdir(file_location)

        with open(f"{file_location}/{file.filename}", "wb+") as f:
            shutil.copyfileobj(file.file, f)

        image = Image.open(f"tmp/{id}/")

        return jsonable_encoder({"message": "ok"})

    except Exception as e:
        return jsonable_encoder({"error": str(e)})


@app.post("/image/ingest/{id}")
async def store_file_with_id(id, file: UploadFile = File(...)):

    try:

        print(file.filename)

        if id == None or id == "":
            raise Exception("Id is required")

        file_location = f"tmp/{id}"

        if not os.path.exists(file_location):
            os.mkdir(file_location)

        with open(f"{file_location}/{file.filename}", "wb+") as f:
            shutil.copyfileobj(file.file, f)

        image = Image.open(f"tmp/{id}/")

        images[id] = image

        return jsonable_encoder({"message": "ok"})

    except Exception as e:

        return jsonable_encoder({"error": str(e)})


@app.delete("/session/{id}")
async def delete_session(id):
    try:
        shutil.rmtree(f"tmp/{id}")
        return jsonable_encoder({"message": "ok"})
    except Exception as e:
        return jsonable_encoder({"error": str(e)})


@app.post("/retriveal/{id}")
async def inference(id, message: Message):

    if id == None or id == "":
        raise Exception("Id is required")

    query = message.content
    text = f"""A chat between a curious user and an artificial intelligence assistant.
    The assistant gives useful, detailed and polite answers to the user's questions.
    The assistant is an expert in the field of analyzing images that people are going to post on their social networks.
    The assistant must respond taking into account the following issues:
    Emotional context of the image:
    Professional context of the image:
    USER: <image>\n{query}
    ASSISTANT:"""
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    image_tensor = model.image_preprocess(images[id])
    output_ids = model.generate(
        input_ids, max_new_tokens=300, images=image_tensor, use_cache=True
    )[0]
    response = tokenizer.decode(
        output_ids[input_ids.shape[1] :], skip_special_tokens=True
    ).strip()

    return response
