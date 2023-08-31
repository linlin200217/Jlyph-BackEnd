#!/usr/bin/python

import pandas as pd
from pprint import pprint
import gensim.downloader
from nltk.stem import WordNetLemmatizer
import nltk
from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionControlNetPipeline, ControlNetModel
import os
import datetime
import random
import numpy as np
import cv2
import rembg


IMAGE_RESOURCE_PATH = "./images"
DATAPATH = "./data"
COLOR = ["pink", "yellow", "green", "blue", "black",
         "grey", "purple", "red", "orange", "white"]
TEXTURE = ["plastic", "iron", "glass", "paper", "fabric",
           "wooden", "stainless steel", "copper", "porcelain", "Rubber"]
PARTIAL_PREFIX = "A "
WHOLE_PREFIX = "A "
COMBINATION_PREDIX = "A "
MASK_CIRCLE = Image.open("images/Mask_Circle.png")
MASK_VERRECTANGLE = Image.open("images/Mask_VerRectangle.png")
MASK_SQUARE = Image.open("images/Mask_Square.png")
MASK = [MASK_CIRCLE, MASK_VERRECTANGLE, MASK_SQUARE]
DEVICE = "cpu"

PIPE_SD = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to(DEVICE)

PIPE_SDCC = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                              controlnet=ControlNetModel.from_pretrained(
                                                                  "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16),
                                                              torch_dtype=torch.float16).to(DEVICE)

def save_image(img: Image.Image, type: str="png"):
    image_id = int(datetime.datetime.now().timestamp())
    img.save(os.path.join(IMAGE_RESOURCE_PATH, f"{image_id}.png", format="png"))
    return image_id
    


def data_process(data_path: str) -> dict:
    data = pd.read_csv(data_path)
    data_title = data_path.split("/")[-1].split(".")[0]
    struct = {
        "data_title": data_title,
        "Categorical": {},
        "Numerical": {}
    }
    for column in data.columns:
        if data[column].dtype.name == "object":
            struct["Categorical"][column] = data[column].tolist()
        else:
            struct["Numerical"][column] = data[column].tolist()
    struct["wordcloud"] = word2vec(data_title)
    return struct


def word2vec(title: str) -> dict:
    glove_vectors = gensim.downloader.load('word2vec-google-news-300')
    corpus_related_words = [word for word,
                            _ in glove_vectors.most_similar(title, topn=20)]
    similarities_dict = {}

    for word in corpus_related_words:
        similarity_score = glove_vectors.similarity(title, word)
        similarities_dict[word] = similarity_score

    max_score = max(similarities_dict.values())
    min_score = min(similarities_dict.values())

    normalized_min = 40
    normalized_max = 100

    normalized_similarities = {}
    for word, score in similarities_dict.items():
        normalized_score = ((score - min_score) / (max_score - min_score)) * \
            (normalized_max - normalized_min) + normalized_min
        normalized_similarities[word] = normalized_score

    lemmatizer = WordNetLemmatizer()

    noun_scores = {}

    for word in corpus_related_words:
        tokens = nltk.word_tokenize(word)
        pos_tags = nltk.pos_tag(tokens)
        for token, pos_tag in pos_tags:
            if pos_tag.startswith('N'):
                singular_word = lemmatizer.lemmatize(token, pos='n')
                if singular_word not in noun_scores:
                    noun_scores[singular_word] = normalized_similarities[word]
    return dict(sorted(noun_scores.items(), key=lambda item: item[1], reverse=True)[:10])


def make_glyph(data: dict) -> str:
    img: Image = PREDESIGN[data['design']](**data)
    image_id = save_image(img)
    return image_id


def make_partial(prompt1: str, guide1: int, *args, **kwargs) -> Image:
    # guide: [0-2]
    prompt = PARTIAL_PREFIX + prompt1
    out = PIPE_SD(prompt=prompt, image=MASK[guide1], strength=0.9, guidance_scale=20)
    return out.images[0]


def make_whole( *args, **kwargs):
    pass


def make_combination( *args, **kwargs):
    pass


PREDESIGN = {
    "partial": make_partial,
    "whole": make_whole,
    "combination": make_combination
}


# regenerate
def generate_glyph(data: dict) -> list[str]:
    df = pd.read_csv(os.path.join(DATAPATH, data['data_title']+".csv"))
    return DESIGN[data['design']](**data)


def make_prompt_by_categorical(prefix: str, prompt: str,categorical:list, df: pd.DataFrame) -> list[str]:
    if len(categorical) == 1:
        item = categorical[0]
        value = item["value"]
        num = len(df[item["column"]].drop_duplicates())
        if value == "color":
            return [f"{prefix}{color} {prompt}" for color in  random.sample(COLOR, num)]
        if value == "texture":
            return [f"{prefix}{texture} {prompt}" for texture in  random.sample(TEXTURE, num)]
    else:
        _tmp = {f"{column1}{column2}" for column1, column2 in zip(df[categorical[0]["column"]], df[categorical[1]["column"]])}
        num = len(_tmp)
        return [f"{prefix}{color} {texture} {prompt}" for color, texture in  zip(random.sample(COLOR, num), random.sample(TEXTURE, num))]


def generate_partial(prompt1: str, Categorical1: list, Numerical: list, df: pd.DataFrame, image_id: str, *args, **kwargs):
    prompts = make_prompt_by_categorical(PARTIAL_PREFIX,prompt1, Categorical1, df)
    return [generate_image(prompt, image_id) for prompt in prompts]

 

def generate_whole( *args, **kwargs):
    pass


def generate_combination( *args, **kwargs):
    pass


def generate_image(prompt: str, image_id: str):
    img = np.array(Image.open(os.path.join(IMAGE_RESOURCE_PATH, image_id+".png")))
    low_threshold = 100
    high_threshold = 200
    img = cv2.Canny(img, low_threshold, high_threshold)
    img = img[:, :, None]
    img = np.concatenate([img, img, img], axis=2)
    canny_image = Image.fromarray(img)

    out = PIPE_SDCC(prompt, num_inference_steps=20, image=canny_image).images[0]
    out = rembg.remove(out)
    out_image_id = save_image(img)


DESIGN = {
    "partial": generate_partial,
    "whole": generate_whole,
    "combination": generate_combination
}
