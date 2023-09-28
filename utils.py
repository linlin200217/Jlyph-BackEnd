import datetime
import math
import os
import random
from typing import Union, List, Dict, Optional, Tuple
import cv2
import gensim.downloader
import nltk
import numpy as np
import rembg
import torch
from PIL import Image, ImageDraw
from diffusers import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
)
from nltk.stem import WordNetLemmatizer

from process import (
    process_to_circle,
    process_to_radiation,
    process_to_transverse,
    process_to_vertical,
    scale_image,
    process_to_combination,
)
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from mpl_toolkits.basemap import Basemap


MODEL_PATH = ""
IMAGE_RESOURCE_PATH = "./resources"
DATAPATH = "./data"
COLOR = [
    "pink",
    "yellow",
    "green",
    "blue",
    "black",
    "grey",
    "purple",
    "red",
    "orange",
    "white",
]
TEXTURE = [
    "plastic",
    "iron",
    "glass",
    "paper",
    "fabric",
    "wooden",
    "stainless steel",
    "copper",
    "porcelain",
    "Rubber",
]
PARTIAL_PREFIX = "A "
WHOLE_PREFIX = "A "
COMBINATION_PREDIX = "A "
MASK_CIRCLE = Image.open("images/Mask_Circle.png")
MASK_VERRECTANGLE = Image.open("images/Mask_VerRectangle.png")
MASK_SQUARE = Image.open("images/Mask_Square.png")
MASK = [MASK_CIRCLE, MASK_VERRECTANGLE, MASK_SQUARE]
DEVICE = "cuda"

os.makedirs(IMAGE_RESOURCE_PATH, exist_ok=True)
os.makedirs(DATAPATH, exist_ok=True)

PIPE_SD = StableDiffusionImg2ImgPipeline.from_pretrained(
    MODEL_PATH + "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to(DEVICE)

PIPE_SDCC = StableDiffusionControlNetPipeline.from_pretrained(MODEL_PATH + "runwayml/stable-diffusion-v1-5",
                                                              controlnet=ControlNetModel.from_pretrained(
                                                                  MODEL_PATH + "lllyasviel/sd-controlnet-canny",
                                                                  torch_dtype=torch.float16),
                                                              torch_dtype=torch.float16).to(DEVICE)

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")


def save_image(img: Image.Image, prefix: Optional[str], type: str = "png") -> str:
    prefix = prefix or "main_"
    image_id = int(datetime.datetime.now().timestamp() +
                   random.randrange(1, 10000))
    filename = f"{prefix}{image_id}.{type}"
    img.save(os.path.join(IMAGE_RESOURCE_PATH, filename), format=type)
    return prefix + str(image_id)


def get_image_by_id(image_id: str) -> Image.Image:
    return Image.open(os.path.join(IMAGE_RESOURCE_PATH, image_id + ".png"))


def data_process(data_path: str) -> Dict:
    data = pd.read_csv(data_path)
    data_title = data_path.split("/")[-1].split(".")[0]
    struct = {"data_title": data_title, "Categorical": {}, "Numerical": {}}
    for column in data.columns:
        if data[column].dtype.name == "object":
            struct["Categorical"][column] = data[column].tolist()
        else:
            struct["Numerical"][column] = data[column].tolist()
    struct["wordcloud"] = word2vec(data_title)
    return struct


def word2vec(title: str) -> Dict:
    glove_vectors = gensim.downloader.load("word2vec-google-news-300")
    corpus_related_words = [
        word for word, _ in glove_vectors.most_similar(title, topn=20)
    ]
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
        normalized_score = ((score - min_score) / (max_score - min_score)) * (
            normalized_max - normalized_min
        ) + normalized_min
        normalized_similarities[word] = normalized_score

    lemmatizer = WordNetLemmatizer()

    noun_scores = {}

    for word in corpus_related_words:
        tokens = nltk.word_tokenize(word)
        pos_tags = nltk.pos_tag(tokens)
        for token, pos_tag in pos_tags:
            if pos_tag.startswith("N"):
                singular_word = lemmatizer.lemmatize(token, pos="n")
                if singular_word not in noun_scores:
                    noun_scores[singular_word] = normalized_similarities[word]
    return dict(
        sorted(noun_scores.items(),
               key=lambda item: item[1], reverse=True)[:10]
    )


def make_glyph(data: Dict) -> List[Dict[str, str]]:
    image_id: List = PREDESIGN[data["design"]](**data)
    return image_id


def make_partial(prompt1: str, guide1: int, img_prefix: Optional[str] = "partial_", *args, **kwargs) -> List[
        Dict[str, str]]:
    # guide: [0-2]
    prompt = PARTIAL_PREFIX + prompt1
    out = PIPE_SD(
        prompt=prompt, image=MASK[int(guide1)], strength=0.9, guidance_scale=20
    )
    return [{"prompt": prompt1, "image_id": save_image(out.images[0], img_prefix)}]


def make_whole(prompt1: Union[str, List], guide1: int, img_prefix: Optional[str] = None, *args, **kwargs) \
        -> List[Dict[str, str]]:
    if isinstance(prompt1, list):
        # if choice sahpe
        return [
            {"prompt": prompt,
             "image_id": save_image(PIPE_SD(prompt=WHOLE_PREFIX + prompt,
                                            image=MASK[int(guide1)], strength=0.9, guidance_scale=20).images[0],
                                    img_prefix)}
            for prompt in prompt1
        ]
    prompt = WHOLE_PREFIX + prompt1
    out = PIPE_SD(
        prompt=prompt, image=MASK[int(guide1)], strength=0.9, guidance_scale=20
    )
    return [{"prompt": prompt1, "image_id": save_image(out.images[0], img_prefix)}]


def make_combination(
    prompt1: Union[str, List], prompt2: str, guide1: int, guide2: int, *args, **kwargs
):
    main_image = make_whole(prompt1, int(guide1))
    sub_image = make_partial(prompt2, int(guide2), "sub_")
    return main_image + sub_image


PREDESIGN = {
    "partial": make_partial,
    "whole": make_whole,
    "combination": make_combination,
}


# generate
def generate_glyph(data: Dict) -> List[str]:
    df = pd.read_csv(os.path.join(DATAPATH, data['data_title'] + ".csv"))
    return DESIGN[data['design']](df=df, **data)


def make_prompt_by_categorical(prefix: str, prompt: str, categorical: List, df: Optional[pd.DataFrame] = None,
                               _color: Optional[List[str]] = None, _texture: Optional[List[str]] = None,
                               _shape: Optional[str] = None, _record=None,
                               _num: Optional[int] = None) -> List[Tuple]:
    # return: (prompt, color, color_value, texture, texture_value, shape, shape_value)
    if _record is None:
        _record = {}
    categorical_pos = ["color", "color_value", "texture",
                       "texture_value", "shape", "shape_value"]
    color_categorical = None
    texture_categorical = None
    shape_categorical = None
    for _categorical in categorical:
        if _categorical["value"] == "color":
            color_categorical = _categorical
        if _categorical["value"] == "texture":
            texture_categorical = _categorical
        if _categorical["value"] == "shape":
            shape_categorical = _categorical
    if len(categorical) == 1:
        item = categorical[0]
        column_duplicated = df[item["column"]].drop_duplicates()
        num = _num or len(column_duplicated)
        if color_categorical:
            return [(f"{prefix}{color} {prompt}", color, value, None, None, None, None)
                    for value, color in zip(column_duplicated, random.sample(_color or COLOR, num))]
        if texture_categorical:
            return [(f"{prefix}{texture} {prompt}", None, None, texture, value, None, None)
                    for value, texture in zip(column_duplicated, random.sample(_texture or TEXTURE, num))]
    elif len(categorical) == 2:
        _tmp = {"c1": {}, "c2": {}, "str": [], "prompt": []}
        num = _num or len({f"{column1}{column2}" for column1, column2 in zip(
            df[categorical[0]["column"]], df[categorical[1]["column"]])})
        _colors = random.sample(_color or COLOR, num)
        _textures = random.sample(_texture or TEXTURE, num)
        for c1, c2 in zip(df[categorical[0]["column"]], df[categorical[1]["column"]]):
            if shape_categorical is None or c2 == _shape:
                s = f"{c1}{c2}"
                if s not in _tmp["str"]:
                    _tmp["c1"][c1] = _record.get(
                        c1) or _tmp["c1"].get(c1) or _colors.pop()
                    _record[c1] = _tmp["c1"][c1]

                    _tmp["c2"][c2] = _record.get(c2) or _tmp["c2"].get(c2) or (_textures.pop()
                                                                               if shape_categorical is None
                                                                               else categorical[1]["column"])
                    _record[c2] = _tmp["c2"][c2]
                    _tmp["str"].append(s)
                    _prompt = [f"A {_tmp['c1'][c1]}" +
                               (f" {_tmp['c2'][c2]}" if shape_categorical is None else "")
                               + f" {prompt}", None, None, None, None, None, None]
                    _prompt[categorical_pos.index(
                        categorical[0]["value"]) + 1] = _tmp['c1'][c1]
                    _prompt[categorical_pos.index(
                        categorical[1]["value"]) + 1] = _tmp['c2'][c2]
                    if categorical[0]["value"] + "_value" in categorical_pos:
                        _prompt[categorical_pos.index(
                            categorical[0]["value"] + "_value") + 1] = c1
                    if categorical[1]["value"] + "_value" in categorical_pos:
                        _prompt[categorical_pos.index(
                            categorical[1]["value"] + "_value") + 1] = c2

                    _tmp["prompt"].append(_prompt)
        return _tmp["prompt"]
    elif len(categorical) == 3:
        _tmp = {"color": {}, "texture": {},
                "shape": {}, "str": [], "prompt": []}
        num = _num or len({f"{column1}{column2}{column3}" for column1, column2, column3 in zip(
            df[categorical[0]["column"]], df[categorical[1]["column"]], df[categorical[2]["column"]])})
        _colors = random.sample(_color or COLOR, num)
        _textures = random.sample(_texture or TEXTURE, num)
        for color, texture, shape in zip(df[categorical[0]["column"]],
                                         df[categorical[1]["column"]],
                                         df[categorical[2]["column"]]):
            if shape == _shape:
                s = f"{color}{texture}{shape}"
                if s not in _tmp["str"]:
                    _tmp["color"][color] = _tmp["color"].get(
                        color) or _colors.pop()
                    _tmp["texture"][texture] = _tmp["texture"].get(
                        texture) or _textures.pop()
                    _tmp["str"].append(s)
                    _tmp["prompt"].append((f"A {_tmp['color'][color]} {_tmp['texture'][texture]} {prompt}",
                                           _tmp['color'][color], color, _tmp['texture'][texture], texture,
                                           categorical[2]["column"], _shape))

        return _tmp["prompt"]


def generate_partial(prompt1: str, Categorical1: List, df: pd.DataFrame, image_id: str,
                     image_prefix: Optional[str] = None, *args, **kwargs):
    prompts = make_prompt_by_categorical(
        PARTIAL_PREFIX, prompt1, Categorical1, df)
    return [{"prompt": prompt1, "color": color, "texture": texture, "color_value": color_value,
             "texture_value": texture_value, "shape": shape, "shape_value": shape_value,
             "image_id": generate_image(prompt, image_id, image_prefix)} for
            prompt, color, color_value, texture, texture_value, shape, shape_value in prompts]


def generate_whole(prompt1: Union[str, List], Categorical1: List, df: pd.DataFrame,
                   image_id: Union[List[Dict], Dict],
                   image_prefix: Optional[str] = "generate_", *args, **kwargs):
    if isinstance(prompt1, list):
        _image_id = []
        _colors = []
        _textures = []
        _record = {}
        for info in image_id:
            # info: {image_id:xx, prompt:xx, shape: xx}
            prompts = make_prompt_by_categorical(WHOLE_PREFIX, info["prompt"], Categorical1, df,
                                                 list(set(COLOR) -
                                                      set(_colors)),
                                                 list(set(TEXTURE) - set(_textures)), info.get("shape"), _record)
            _textures += [t for _, _, _, t, _, _, _ in prompts]
            _colors += [c for _, c, _, _, _, _, _ in prompts]
            _image_id += [{"prompt": prompt, "color": color, "color_value": color_value, "texture": texture,
                           "texture_value": texture_value, "shape": shape, "shape_value": shape_value,
                           "image_id": generate_image(prompt, info["image_id"], image_prefix)}
                          for prompt, color, color_value, texture, texture_value, shape, shape_value in prompts]
        return _image_id
    prompts = make_prompt_by_categorical(
        WHOLE_PREFIX, prompt1, Categorical1, df, _shape=image_id.get("shape"))
    return [{"prompt": prompt1, "color": color, "texture": texture, "color_value": color_value,
             "texture_value": texture_value, "shape": shape, "shape_value": shape_value,
             "image_id": generate_image(prompt, image_id["image_id"], image_prefix)}
            for prompt, color, color_value, texture, texture_value, shape, shape_value in prompts]


def generate_combination(image_id: List, prompt1: Union[str, List], prompt2: str, Categorical1: List,
                         Categorical2: List, df: pd.DataFrame, *args, **kwargs):
    main_images = generate_whole(
        prompt1, Categorical1, df, image_id[:-1], "main_generated_") \
        if isinstance(prompt1, List) else generate_partial(prompt1, Categorical1, df, image_id[0].get("image_id"),
                                                           "main_generated_")
    sub_image = generate_partial(
        prompt2, Categorical2, df, image_id[-1]["image_id"], "sub_generated_")
    return main_images + sub_image


def generate_image(prompt: str, image_id: str, image_prefix: Optional[str] = None):
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
    "combination": generate_combination,
}


def regenerate_by_prompt(
    image_id: str,
    design: str,
    prompt: str,
    color: Optional[str] = None,
    texture: Optional[str] = None,
):
    prefix = (
        PARTIAL_PREFIX
        if design == "partial"
        else WHOLE_PREFIX
        if design == "whole"
        else COMBINATION_PREDIX
    )
    regenerate_prompt = f"{prefix}{f'{color} {texture}' if color and texture else color if color else texture} {prompt}"
    image_id = generate_image(
        regenerate_prompt, image_id, "_".join(image_id.split("_")[:-1])+"_"
    )
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
                      color_column: Optional[str],
                      texture_column: Optional[str],
                      numerical: Dict,
                      df: pd.DataFrame,
                      process_type: str,
                      image_pipe: List[List],
                      *args, **kwargs):
    data = df[(color is None or df[color_column] == color) & (
        texture is None or df[texture_column] == texture)]
    column_name = numerical["column"]
    column_value = numerical["value"]
    if column_name == "number":
        for column_idx, number in zip(data.index, data[column_value]):
            image_pipe.append([column_idx, scale_image(process_to_radiation(image, number))
                               if process_type else process_to_circle(image, number)])
    if column_name == "size":
        for idx, (column_idx, size) in enumerate(zip(data.index, data[column_value])):
            image_width, image_height = scale_size(
                df[column_value].max(), df[column_value].min(), size, image)
            if image_pipe:
                image_pipe[idx][1] = image_pipe[idx][1].resize(
                    (image_width, image_height))
            else:
                image_pipe.append(
                    [column_idx, image.resize((image_width, image_height))])
    if column_name == "opacity":
        for idx, (column_idx, opacity) in enumerate(zip(data.index, data[column_value])):
            if image_pipe:
                image_pipe[idx][1] = set_alpha(image_pipe[idx][1],
                                               calculate_opacity(df[column_value].max(),
                                                                 df[column_value].min(
                                               ),
                    opacity))
            else:
                image_pipe.append([column_idx,
                                   set_alpha(image,
                                             calculate_opacity(df[column_value].max(),
                                                               df[column_value].min(
                                             ),
                                                 opacity))])


def numerical_whole(image: Image.Image,
                    color: Optional[str],
                    texture: Optional[str],
                    color_column: Optional[str],
                    texture_column: Optional[str],
                    shape: Optional[str],
                    shape_column: Optional[str],
                    size_of_whole: int,
                    numerical: str,
                    df: pd.DataFrame,
                    process_type: int,
                    image_pipe: List[List],
                    *args, **kwargs):
    data = df[(color is None or df[color_column] == color) & (texture is None or df[texture_column] == texture)
              & (shape is None or df[shape_column] == shape)]
    column_name = numerical["column"]
    column_value = numerical["value"]
    empty_pipe = image_pipe == []
    if column_name == "number":
        for idx, (column_idx, number) in enumerate(zip(data.index, data[column_value])):
            if empty_pipe:
                image_pipe.append([column_idx, process_to_transverse(image, size_of_whole, number) if process_type
                                   else process_to_vertical(image, size_of_whole, number)])
            else:
                image_pipe[idx][1] = process_to_transverse(image_pipe[idx][1], size_of_whole, number) if process_type \
                    else process_to_vertical(image_pipe[idx][1], size_of_whole, number)
    if column_name == "size":
        for idx, (column_idx, size) in enumerate(zip(data.index, data[column_value])):
            image_width, image_height = scale_size(
                df[column_value].max(), df[column_value].min(), size, image)
            if empty_pipe:
                image_pipe.append(
                    [column_idx, image.resize((image_width, image_height))])
            else:
                image_pipe[idx][1] = image_pipe[idx][1].resize(
                    (image_width, image_height))
    if column_name == "opacity":
        for idx, (column_idx, opacity) in enumerate(zip(data.index, data[column_value])):
            if empty_pipe:
                image_pipe.append([column_idx, set_alpha(image,
                                                         calculate_opacity(df[column_value].max(),
                                                                           df[column_value].min(
                                                         ),
                                                             opacity))])
            else:
                image_pipe[idx][1] = set_alpha(image_pipe[idx][1],
                                                calculate_opacity(df[column_value].max(),
                                                                  df[column_value].min(
                                                ),
                                          opacity))


def numerical_combination(image: Image.Image,
                          color: Optional[str],
                          texture: Optional[str],
                          color_column: Optional[str],
                          texture_column: Optional[str],
                          shape: Optional[str],
                          shape_column: Optional[str],
                          df: pd.DataFrame,
                          process_type: int,
                          numerical: Dict,
                          sub_image: Dict,
                          image_pipe: List[List],
                          numerical_number: Dict,
                          *args, **kwargs):
    # print(f"""{color_column}: {color}, {sub_image.get("color_column")}:{sub_image.get("color")},
    #     {texture_column}: {texture}, {sub_image.get("texture_column")}:{sub_image.get("texture")},
    #     {shape_column}:{shape}""")
    try:
        data = df[(color is None or df[color_column] == color) & (texture is None or df[texture_column] == texture)
                  & (shape is None or df[shape_column] == shape)
                  & (sub_image.get("color") is None or df[sub_image["color_column"]] == sub_image["color"])
                  & (sub_image.get("texture") is None or df[sub_image["texture_column"]] == sub_image["texture"])]
    except KeyError:
        print("KEYERROR")
        return
    if data.empty:
        print("EMPTY DATAS")
        return
    # print(image_pipe)
    sub_image = get_image_by_id(sub_image["image_id"])
    column_name = numerical["column"]
    column_value = numerical["value"]
    if column_name == "number1" and process_type:
        for column_idx, sub_of_number, main_number in zip(data.index, data[numerical_number["value"]],
                                                          data[column_value]):
            main_image = scale_image(process_to_radiation(image, main_number)) if process_type == 1 \
                else process_to_circle(image, main_number)
            image_pipe.append([column_idx, process_to_combination(
                main_image, sub_image, sub_of_number)])
    if column_name == "number" and process_type == 0:
        for column_idx, number in zip(data.index, data[column_value]):
            image_pipe.append(
                [column_idx, process_to_combination(image, sub_image, number)])
    if column_name == "size":
        for idx, (column_idx, size) in enumerate(zip(data.index, data[column_value])):
            if image_pipe:
                image_pipe[idx][1] = image_pipe[idx][1].resize(
                    scale_size(df[column_value].max(), df[column_value].min(), size, image))
            else:
                image_pipe.append(
                    [column_idx,
                     process_to_combination(
                         resize_image_of_combination(image, size),
                         sub_image,
                         df[column_value][idx]
                     )]
                )
    if column_name == "opacity":
        for idx, (column_idx, opacity) in enumerate(zip(data.index, data[column_value])):
            if image_pipe:
                image_pipe[idx][1] = set_alpha(image_pipe[idx][1],
                                               calculate_opacity(df[column_value].max(),
                                                                 df[column_value].min(
                                               ),
                    opacity))
            else:
                image_pipe.append([column_idx, set_alpha(image,
                                                         calculate_opacity(df[column_value].max(),
                                                                           df[column_value].min(
                                                         ),
                                                             opacity))])


PROCESS = {
    "partial": numerical_partial,
    "whole": numerical_whole,
    "combination": numerical_combination,
}


def process_image_by_numerical(data: Dict):
    data_title = data["data_title"]
    df = pd.read_csv(os.path.join(DATAPATH, data_title + ".csv"))
    images = data.get("images")
    Numerical = data.get("Numerical")
    design = data.get("design")
    sub_images = [item for item in images if "sub_" in item["image_id"]
                  ] if design == "combination" else None
    main_images = [item for item in images if "main_" in item["image_id"]
                   ] if design == "combination" else images

    result_images = []  # [(idx_of_dataform, <Image>)....]
    for item in main_images:
        image_id = item.get("image_id")
        color = item.get("color")
        texture = item.get("texture")
        shape = item.get("shape")
        color_column = item.get("color_column")
        texture_column = item.get("texture_column")
        shape_column = item.get("shape_column")

        image = get_image_by_id(image_id)
        image_pipe = []

        numericals = [numerical["column"] for numerical in Numerical]
        numerical_number = Numerical[numericals.index(
            "number")] if "number" in numericals else None

        if sub_images:
            for sub_image in sub_images:
                for numerical in Numerical:
                    PROCESS[design](image=image, color=color, texture=texture, df=df, color_column=color_column,
                                    texture_column=texture_column, numerical=numerical, shape=shape,
                                    shape_column=shape_column, image_pipe=image_pipe, sub_image=sub_image,
                                    numerical_number=numerical_number,
                                    **data)

        else:
            for numerical in Numerical:
                PROCESS[design](image=image, color=color, texture=texture, df=df, color_column=color_column,
                                texture_column=texture_column, numerical=numerical, shape=shape,
                                shape_column=shape_column, image_pipe=image_pipe, sub_image=None,
                                numerical_number=numerical_number, **data)
        result_images += image_pipe
    return [(column_index, save_image(img, "res_")) for column_index, img in result_images]


def scale_size(max_of_data, min_of_data, size, image) -> Tuple[int, int]:
    scaling = 0.5 + 0.5 * (size - min_of_data) / (max_of_data - min_of_data)

    image_width = int(image.size[0] * scaling)
    image_height = int(image.size[1] * scaling)
    return image_width, image_height


def calculate_opacity(max_of_opacity, min_of_opacity, opacity):
    return 25 + ((opacity - min_of_opacity) / (max_of_opacity - min_of_opacity)) * (100 - 25)


def resize_image_of_combination(image, scale_factor):
    original_width, original_height = image.size

    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    resized_dog_image = image.resize((new_width, new_height), Image.ANTIALIAS)

    background = Image.new(
        'RGBA', (original_width, original_height), (0, 0, 0, 0))

    paste_x = (original_width - new_width) // 2
    paste_y = (original_height - new_height) // 2

    background.paste(resized_dog_image, (paste_x, paste_y), resized_dog_image)

    return background


def set_alpha(img, alpha_percentage) -> Image.Image:
    assert 0 <= alpha_percentage <= 100, "Alpha percentage should be between 0 and 100."

    img = img.convert("RGBA")

    datas = img.getdata()

    new_data = []
    for item in datas:
        # 修改alpha值
        new_data.append(
            (item[0], item[1], item[2], int(item[3] * alpha_percentage / 100))
        )

    img.putdata(new_data)

    return img


def draw_dashed_line(
    draw, start, end, fill="black", width=1, dash_length=10, space_length=5
):
    """绘制虚线"""
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    length = dash_length + space_length
    dash_count = int(math.sqrt(dx**2 + dy**2) // length)

    for i in range(dash_count):
        start_x = x1 + dx * (i / dash_count)
        start_y = y1 + dy * (i / dash_count)
        end_x = x1 + dx * ((i + 1) / dash_count)
        end_y = y1 + dy * ((i + 1) / dash_count)
        if i % 2 == 0:  # Only draw every other segment to create dashed effect
            draw.line([(start_x, start_y), (end_x, end_y)],
                      fill=fill, width=width)


def make_grid(
        images: List[Dict],
        border_thickness: int,
        color_fill: str,
        dashed: bool,
        draw_lines: bool,
        background_type: str,
        background_color: str,
        *args, **kwargs) -> str:
    def center_crop(image, target_width, target_height):
        center_x = image.width // 2
        center_y = image.height // 2

        left = center_x - target_width // 2
        top = center_y - target_height // 2
        right = center_x + target_width // 2
        bottom = center_y + target_height // 2

        return image.crop((left, top, right, bottom))

    flower_images = [(image.get("data_index"), get_image_by_id(
        image.get("image_id"))) for image in images]
    flower_images.sort(key=lambda item: item[0])
    N = math.ceil(math.sqrt(len(flower_images)))

    output_width = N * 500
    output_height = N * 500

    output_image = Image.new('RGBA', (output_width, output_height), (255, 255, 255, 0)) \
        if background_type == "transparent" else \
        Image.new('RGBA', (output_width, output_height), background_color)

    for idx, (column_index, flower_image) in enumerate(flower_images):
        row = idx // N
        col = idx % N

        cropped_flower = center_crop(flower_image, 500, 500)
        x_offset = col * 500
        y_offset = row * 500
        output_image.paste(
            cropped_flower, (x_offset, y_offset), cropped_flower)

    draw = ImageDraw.Draw(output_image)

    if draw_lines:
        for i in range(1, N ** 2):
            x = i * 500
            if dashed:
                draw_dashed_line(
                    draw,
                    (x, 0),
                    (x, output_height),
                    fill=color_fill,
                    width=border_thickness,
                )
            else:
                draw.line(
                    [(x, 0), (x, output_height)],
                    fill=color_fill,
                    width=border_thickness,
                )

    # 绘制水平线
    if draw_lines:
        for i in range(1, N ** 2):
            y = i * 500
            if dashed:
                draw_dashed_line(
                    draw,
                    (0, y),
                    (output_width, y),
                    fill=color_fill,
                    width=border_thickness,
                )
            else:
                draw.line(
                    [(0, y), (output_width, y)], fill=color_fill, width=border_thickness
                )

    return save_image(output_image.resize((1024, 1024)), "placement_")


def make_struct(
        images: List[Dict],
        data_title: str,
        column1: str,
        column2: str,
        background_type: str,
        background_color: str,
        canvas_color: str,
        text_size: int,
        text_color: str,
        grid_color: str,
        show_grid: bool,
        *args, **kwargs
) -> str:
    df = pd.read_csv(os.path.join(DATAPATH, data_title + ".csv"))
    fig, ax = plt.subplots(figsize=(5.12, 5.12), dpi=500)

    images = [(image.get("data_index"), get_image_by_id(
        image.get("image_id"))) for image in images]

    if background_type == 'transparent':
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
    elif background_type == "color":
        fig.patch.set_facecolor(canvas_color)
        ax.set_facecolor(background_color)

    for column_index, image in images:
        x_data = df[column1][column_index]
        y_data = df[column2][column_index]
        imagebox = OffsetImage(image, zoom=0.15)  # 这里的zoom可以调整
        ab = AnnotationBbox(imagebox, (x_data, y_data), frameon=False)
        ax.add_artist(ab)

    # for x_data, y_data, img in zip(df[column1], df[column2], images):
    #     imagebox = OffsetImage(img, zoom=0.15)  # 这里的zoom可以调整
    #     ab = AnnotationBbox(imagebox, (x_data, y_data), frameon=False)
    #     ax.add_artist(ab)

    ax.set_xlabel(column1, fontsize=text_size, color=text_color, weight='bold')
    ax.set_ylabel(column2, fontsize=text_size, color=text_color, weight='bold')
    ax.scatter(df[column1], df[column2], alpha=0)  # 透明的散点，只为确定轴的范围
    # Calculate the new range for x-axis and y-axis
    x_range = df[column1].max() - df[column1].min()
    y_range = df[column2].max() - df[column2].min()

    # Calculate the new limits by adding/subtracting 10% of the range
    x_min = df[column1].min() - 0.2 * x_range
    x_max = df[column1].max() + 0.2 * x_range
    y_min = df[column2].min() - 0.2 * y_range
    y_max = df[column2].max() + 0.2 * y_range

    # Set the new x and y axis limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # 设置轴的颜色
    ax.spines["bottom"].set_color(text_color)
    ax.spines["top"].set_color(text_color)
    ax.spines["left"].set_color(text_color)
    ax.spines["right"].set_color(text_color)
    ax.tick_params(axis="both", colors=text_color,
                   which="major", labelsize=text_size)

    # 显示网格
    if show_grid:
        ax.grid(True, which="both", linestyle="--",
                linewidth=1.5, color=grid_color)
        ax.set_axisbelow(True)

    # 保存图形
    plt.tight_layout()
    filename = f'placement_{int(datetime.datetime.now().timestamp())}'
    plt.savefig(os.path.join(IMAGE_RESOURCE_PATH, filename + ".png"), dpi=500, bbox_inches='tight', pad_inches=0,
                transparent=True if background_type == 'transparent' else False)
    return filename


def make_geo(
        images: List[Dict],
        data_title: str,
        column1: str,
        column2: str,
        fill_color: str,
        continent_color: str,
        countries_color: str,
        linestyle: str,
        coastlines: str,
        lake_color: str,
        *args, **kwargs

) -> str:
    data = pd.read_csv(os.path.join(DATAPATH, data_title + ".csv"))
    images = [(image.get("data_index"), get_image_by_id(
        image.get("image_id"))) for image in images]

    lats = data[column1]
    lons = data[column2]

    fig, ax = plt.subplots(figsize=(5.12, 5.12), dpi=500)
    m = Basemap(
        projection="merc",
        resolution="i",
        llcrnrlon=min(lons) - 5,
        urcrnrlon=max(lons) + 5,
        llcrnrlat=min(lats) - 5,
        urcrnrlat=max(lats) + 5,
        ax=ax,
    )

    m.drawmapboundary(fill_color=fill_color, linewidth=0)
    m.fillcontinents(color=continent_color, lake_color=lake_color)
    m.drawcountries(linewidth=2, linestyle=linestyle, color=countries_color)
    m.drawcoastlines(linewidth=0.5, color=coastlines)

    # 在地图上添加城市的图片
    for column_index, image in images:
        lat = data[column1][column_index]
        lon = data[column2][column_index]
        x, y = m(lon, lat)
        size_factor = 0.1  # 使图片大小与年份相关
        img = OffsetImage(image, zoom=size_factor)
        ax.add_artist(AnnotationBbox(img, (x, y), frameon=False))

    # for lat, lon, image in zip(lats, lons, images):
    #     x, y = m(lon, lat)
    #     size_factor = 0.1  # 使图片大小与年份相关
    #     img = OffsetImage(image, zoom=size_factor)
    #     ax.add_artist(AnnotationBbox(img, (x, y), frameon=False))

    filename = f'placement_{int(datetime.datetime.now().timestamp())}'
    plt.savefig(os.path.join(IMAGE_RESOURCE_PATH, filename + ".png"), dpi=500)
    return filename


def make_drawer_by_height(images: List[Dict],
                          draw_lines: bool,
                          line_width: int,
                          line_color: str,
                          # dashed: bool,
                          background_color: Optional[str] = None,
                          *args, **kwargs) -> str:
    images = [get_image_by_id(image_id.get("image_id")) for image_id in images]
    total_height = sum(img.height for img in images)

    result_image = Image.new(
        "RGBA", (total_height, total_height), background_color)
    y_offset = 0

    for i, img in enumerate(images):
        result_image.paste(img, (0, y_offset), img)
        if draw_lines:
            draw = ImageDraw.Draw(result_image)
            if draw_lines:
                draw.line(
                    [(0, y_offset), (total_height, y_offset)],
                    fill=line_color,
                    width=line_width,
                )

            y_offset += img.height
            y_offset += line_width
    # return save_image(Image.frombytes("RGBA", (total_height, total_height),result_image.tobytes()), "placement_")
    return save_image(result_image, "placement_")


def make_drawer_by_width(images: List[Dict],
                         draw_lines: bool,
                         line_width: int,
                         line_color: str,
                         background_color: Optional[str] = None,
                         *args, **kwargs):
    images = [get_image_by_id(image_id.get("image_id")) for image_id in images]
    total_width = sum(img.width for img in images)

    result_image = Image.new(
        "RGBA", (total_width, total_width), background_color)
    x_offset = 0

    for i, img in enumerate(images):
        result_image.paste(img, (x_offset, 0), img)
        draw = ImageDraw.Draw(result_image)

        if draw_lines:
            draw.line(
                [(x_offset, 0), (x_offset, total_width)],
                fill=line_color,
                width=line_width,
            )
        x_offset += img.width
        x_offset += line_width

    return result_image


def placement_partial_combination(data: Dict) -> str:
    method = data.get("method")
    if method == "grid":
        return make_grid(**data)
    if method == "struct":
        return make_struct(**data)
    if method == "geo":
        return make_geo(**data)


def placement_whole(data: Dict):
    method = data.get("method")
    if method == "grid":
        return (
            make_drawer_by_height(**data)
            if data.get("drawer_by") == "height"
            else make_drawer_by_width(**data)
        )
    if method == "struct":
        return make_struct(**data)
    if method == "geo":
        return make_geo(**data)


def placement(data):
    return PLACEMENT[data.get("design")](data)


PLACEMENT = {
    "partial": placement_partial_combination,
    "whole": placement_whole,
    "combination": placement_partial_combination,
}
