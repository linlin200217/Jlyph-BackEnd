import datetime
import math
import os
import random
from typing import Union, List, Dict, Optional, Tuple

import cv2
import gensim.downloader
import nltk
import numpy as np
import pandas
import pandas as pd
import rembg
import torch
from PIL import Image, ImageDraw
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionControlNetPipeline, ControlNetModel
from nltk.stem import WordNetLemmatizer

from process import (process_to_circle,
                     process_to_radiation,
                     process_to_transverse,
                     process_to_vertical,
                     scale_image,
                     process_to_combination)

IMAGE_RESOURCE_PATH = "./resources"
DATAPATH = "./data"
COLOR = ["pink", "yellow", "green", "blue", "black",
         "grey", "purple", "red", "orange", "white"]
TEXTURE = ["plastic", "iron", "glass", "paper", "fabric",
           "wooden", "stainless steel", "copper", "porcelain", "Rubber"]
PARTIAL_PREFIX = "A "
WHOLE_PREFIX = "A "
COMBINATION_PREDIX = "A "
MASK_CIRCLE = Image.open("src/Mask_Circle.png")
MASK_VERRECTANGLE = Image.open("src/Mask_VerRectangle.png")
MASK_SQUARE = Image.open("src/Mask_Square.png")
MASK = [MASK_CIRCLE, MASK_VERRECTANGLE, MASK_SQUARE]
DEVICE = "cpu"

os.makedirs(IMAGE_RESOURCE_PATH, exist_ok=True)
os.makedirs(DATAPATH, exist_ok=True)

PIPE_SD = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to(DEVICE)

PIPE_SDCC = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                              controlnet=ControlNetModel.from_pretrained(
                                                                  "lllyasviel/sd-controlnet-canny",
                                                                  torch_dtype=torch.float16),
                                                              torch_dtype=torch.float16).to(DEVICE)


def save_image(img: Image.Image, prefix: Optional[str], type: str = "png") -> str:
    prefix = prefix or "main_"
    image_id = int(datetime.datetime.now().timestamp())
    filename = f"{prefix}{image_id}.{type}"
    img.save(os.path.join(IMAGE_RESOURCE_PATH,
                          filename), format=type)
    return prefix + str(image_id)


def get_image_by_id(image_id: str) -> Image.Image:
    return Image.open(os.path.join(IMAGE_RESOURCE_PATH, image_id + ".png"))


def data_process(data_path: str) -> Dict:
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


def word2vec(title: str) -> Dict:
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


def make_glyph(data: Dict) -> List[Dict[str, str]]:
    image_id: List = PREDESIGN[data['design']](**data)
    return image_id


def make_partial(prompt1: str, guide1: int, img_prefix: Optional[str], *args, **kwargs) -> List[Dict[str, str]]:
    # guide: [0-2]
    prompt = PARTIAL_PREFIX + prompt1
    out = PIPE_SD(
        prompt=prompt, image=MASK[guide1], strength=0.9, guidance_scale=20)
    return [{"prompt": prompt1, "image_id": save_image(out.images[0], img_prefix)}]


def make_whole(prompt1: Union[str, List], guide1: int, img_prefix: Optional[str] = None, *args, **kwargs) -> List[
    Dict[str, str]]:
    if isinstance(prompt1, list):
        # if choice sahpe
        return [
            {"prompt": prompt,
             "image_id": save_image(PIPE_SD(prompt=WHOLE_PREFIX + prompt,
                                            image=MASK[guide1], strength=0.9, guidance_scale=20).images[0], img_prefix)}
            for prompt in prompt1
        ]
    prompt = WHOLE_PREFIX + prompt1
    out = PIPE_SD(
        prompt=prompt, image=MASK[guide1], strength=0.9, guidance_scale=20)
    return [{"prompt": prompt1, "image_id": save_image(out.images[0], img_prefix)}]


def make_combination(prompt1: Union[str, List], prompt2: str, guide1: int, guide2: int, *args, **kwargs):
    main_image = make_whole(prompt1, guide1)
    sub_image = make_partial(prompt2, guide2, "sub_")
    return main_image + sub_image


PREDESIGN = {
    "partial": make_partial,
    "whole": make_whole,
    "combination": make_combination
}


# generate
def generate_glyph(data: Dict) -> List[str]:
    df = pd.read_csv(os.path.join(DATAPATH, data['data_title'] + ".csv"))
    return DESIGN[data['design']](df=df, **data)


def make_prompt_by_categorical(prefix: str, prompt: str, categorical: List, df: Optional[pd.DataFrame] = None,
                               _color: Optional[List[str]] = None, _texture: Optional[List[str]] = None,
                               _num: Optional[int] = None) -> List[Tuple]:
    if len(categorical) == 1:
        item = categorical[0]
        value = item["value"]
        num = _num or len(df[item["column"]].drop_duplicates())
        if value == "color":
            return [(f"{prefix}{color} {prompt}", color, None) for color in random.sample(_color or COLOR, num)]
        if value == "texture":
            return [(f"{prefix}{texture} {prompt}", None, texture) for texture in
                    random.sample(_texture or TEXTURE, num)]
    else:
        num = _num or len({f"{column1}{column2}" for column1, column2 in zip(
            df[categorical[0]["column"]], df[categorical[1]["column"]])})
        return [(f"{prefix}{color} {texture} {prompt}", color, texture) for color, texture in
                zip(random.sample(_color or COLOR, num), random.sample(_texture or TEXTURE, num))]


def generate_partial(prompt1: str, Categorical1: List, Numerical: List, df: pd.DataFrame, image_id: str,
                     image_prefix: Optional[str] = None, *args, **kwargs):
    prompts = make_prompt_by_categorical(
        PARTIAL_PREFIX, prompt1, Categorical1, df)
    return [{"prompt": prompt1, "color": color, "texture": texture,
             "image_id": generate_image(prompt, image_id, image_prefix)} for prompt, color, texture in prompts]


def generate_whole(prompt1: Union[str, List], Categorical1: List, Numerical: List, df: pd.DataFrame, image_id: str,
                   image_prefix: Optional[str] = None, *args, **kwargs):
    if isinstance(prompt1, list):
        _image_id = []
        for prompt in prompt1:
            prompts = make_prompt_by_categorical(WHOLE_PREFIX, prompt, Categorical1, df)
            _image_id += [{"prompt": prompt, "color": color, "texture": texture,
                           "image_id": generate_image(prompt, image_id, image_prefix)} for prompt, color, texture in
                          prompts]
        return _image_id
    prompts = make_prompt_by_categorical(
        WHOLE_PREFIX, prompt1, Categorical1, df)
    return [{"prompt": prompt1, "color": color, "texture": texture,
             "image_id": generate_image(prompt, image_id, image_prefix)} for prompt, color, texture in prompts]


def generate_combination(image_id: List, prompt1: Union[str, List], prompt2: str, Categorical1: List,
                         Categorical2: List, df: pd.DataFrame, Numerical: List, *args, **kwargs):
    main_images = generate_whole(
        prompt1, Categorical1, Numerical, df, image_id[:-1], "main_generated_")
    sub_image = generate_partial(
        prompt2, Categorical2, Numerical, df, image_id[-1], "sub_generated_")
    return main_images + sub_image


def generate_image(prompt: str, image_id: str, image_prefix: None = None):
    img = np.array(get_image_by_id(image_id))
    low_threshold = 100
    high_threshold = 200
    img = cv2.Canny(img, low_threshold, high_threshold)
    img = img[:, :, None]
    img = np.concatenate([img, img, img], axis=2)
    canny_image = Image.fromarray(img)

    out = PIPE_SDCC(prompt, num_inference_steps=20,
                    image=canny_image).images[0]
    out = rembg.remove(out)
    out_image_id = save_image(out, image_prefix)
    return out_image_id


DESIGN = {
    "partial": generate_partial,
    "whole": generate_whole,
    "combination": generate_combination
}


def regenerate_by_prompt(image_id: str, design: str, prompt: str, color: Optional[str] = None,
                         texture: Optional[str] = None):
    prefix = PARTIAL_PREFIX if design == "partial" else WHOLE_PREFIX if design == "whole" else COMBINATION_PREDIX
    regenerate_prompt = f"{prefix}{f'{color} {texture}' if color and texture else color if color else texture} {prompt}"
    image_id = generate_image(regenerate_prompt, image_id)
    return {
        "image_id": image_id,
        "prompt": prompt,
        "texture": texture,
        "color": color,
    }


# image process

def numerical_partial(image: Image.Image,
                      color: Optional[str],
                      texture: Optional[str],
                      numerical: str,
                      df: pandas.DataFrame,
                      process_type: str,
                      image_pipe: List,
                      *args, **kwargs):
    data = df[(color is None or df["color"] == color) & (texture is None or df["texture"] == texture)]
    if numerical == "number":
        for number in data[numerical]:
            image_pipe.append(scale_image(process_to_radiation(image, number)) if process_type
                              else process_to_circle(image, number))
    if numerical == "size":
        pass


def numerical_whole(image: Image.Image,
                    color: Optional[str],
                    texture: Optional[str],
                    size_of_whole: int,
                    numerical: str,
                    df: pandas.DataFrame,
                    process_type: int,
                    image_pipe: List,
                    *args, **kwargs):
    data = df[(color is None or df["color"] == color) & (texture is None or df["texture"] == texture)]
    if numerical == "number":
        for number in data[numerical]:
            image_pipe.append(process_to_transverse(image, size_of_whole, number) if process_type
                              else process_to_vertical(image, size_of_whole, number))
    if numerical == "size":
        pass


def numerical_combination(image: Image.Image,
                          color: Optional[str],
                          texture: Optional[str],
                          df: pandas.DataFrame,
                          process_type: int,
                          numerical: str,
                          sub_image: Image.Image,
                          image_pipe: List,
                          *args, **kwargs):
    data = df[(color is None or df["color"] == color) & (texture is None or df["texture"] == texture)]
    if process_type:
        if numerical == "number1":
            for sub_of_number, number in zip(data["number"], data[numerical]):
                main_image = scale_image(process_to_radiation(image, number)) if process_type else process_to_circle(
                    image, number)
                image_pipe.append(process_to_combination(main_image, sub_image, sub_of_number, (250, 250), 200))

        if numerical == "size":
            pass


PROCESS = {
    "partial": numerical_partial,
    "whole": numerical_whole,
    "combination": numerical_combination
}


def process_image_by_numerical(data: Dict):
    data_title = data["data_title"]
    df = pd.read_csv(os.path.join(DATAPATH, data_title + ".csv"))
    images = data.get("images")
    Numerical = data.get("Numerical")
    design = data.get("design")
    sub_image = get_image_by_id(images[-1]) if design == "combination" else None
    images = images[:-1] if design == "combination" else images

    for item in images:
        image_id = item.get("image_id")
        color = item.get("color")
        texture = item.get("texture")

        image = get_image_by_id(image_id)
        image_pipe = []
        for numerical in Numerical:
            PROCESS[design](image=image, color=color, texture=texture, df=df, numerical=numerical,
                            image_pipe=image_pipe, sub_image=sub_image, **data)
