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
    "/home/newdisk/Website/stable-diffusion-v1-5", torch_dtype=torch.float16
).to(DEVICE)

controlnet = ControlNetModel.from_pretrained(
        "/home/newdisk/Website/sd-controlnet-canny", torch_dtype=torch.float16
    )

PIPE_SDCC = StableDiffusionControlNetPipeline.from_pretrained(
    "/home/newdisk/Website/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
).to(DEVICE)


nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")


def save_image(img: Image.Image, prefix: Optional[str], type: str = "png") -> str:
    prefix = prefix or "main_"
    image_id = int(datetime.datetime.now().timestamp() + random.randrange(1, 10000))
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
        sorted(noun_scores.items(), key=lambda item: item[1], reverse=True)[:10]
    )


def make_glyph(data: Dict) -> List[Dict[str, str]]:
    image_id: List = PREDESIGN[data["design"]](**data)
    return image_id


def make_partial(
    prompt1: str, guide1: int, img_prefix: Optional[str] = "partial_", *args, **kwargs
) -> List[Dict[str, str]]:
    # guide: [0-2]
    prompt = PARTIAL_PREFIX + prompt1
    out = PIPE_SD(
        prompt=prompt, image=MASK[int(guide1)], strength=0.9, guidance_scale=20
    )
    return [{"prompt": prompt1, "image_id": save_image(out.images[0], img_prefix)}]


def make_whole(
    prompt1: Union[str, List],
    guide1: int,
    img_prefix: Optional[str] = None,
    *args,
    **kwargs,
) -> List[Dict[str, str]]:
    if isinstance(prompt1, list):
        # if choice sahpe
        return [
            {
                "prompt": prompt,
                "image_id": save_image(
                    PIPE_SD(
                        prompt=WHOLE_PREFIX + prompt,
                        image=MASK[int(guide1)],
                        strength=0.9,
                        guidance_scale=20,
                    ).images[0],
                    img_prefix,
                ),
            }
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
    df = pd.read_csv(os.path.join(DATAPATH, data["data_title"] + ".csv"))
    return DESIGN[data["design"]](df=df, **data)


def make_prompt_by_categorical(
    prefix: str,
    prompt: str,
    categorical: List,
    df: Optional[pd.DataFrame] = None,
    _color: Optional[List[str]] = None,
    _texture: Optional[List[str]] = None,
    _num: Optional[int] = None,
) -> List[Tuple]:
    if len(categorical) == 1:
        item = categorical[0]
        value = item["value"]
        num = _num or len(df[item["column"]].drop_duplicates())
        if value == "color":
            return [
                (f"{prefix}{color} {prompt}", color, None)
                for color in random.sample(_color or COLOR, num)
            ]
        if value == "texture":
            return [
                (f"{prefix}{texture} {prompt}", None, texture)
                for texture in random.sample(_texture or TEXTURE, num)
            ]
    else:
        _tmp = {"color":[], "texture": [], "str":[]}
        num = _num or len({f"{column1}{column2}" for column1, column2 in zip(
            df[categorical[0]["column"]], df[categorical[1]["column"]])})
        _colors = random.sample(_color or COLOR, num)
        _textures = random.sample(_texture or TEXTURE, num)
        for color, texture in zip(df[categorical[0]["column"]], df[categorical[1]["column"]]):
            s = f"{color}{texture}"
            if s not in _tmp["str"]:
                _tmp["color"].append(_colors.pop())
                _tmp["texture"].append(_textures.pop())
                _tmp["str"].append(s)

        return [(f"{prefix}{color} {texture} {prompt}", color, texture) for color, texture in zip(_tmp["color"], _tmp["color"])]


def generate_partial(prompt1: str, Categorical1: List, Numerical: List, df: pd.DataFrame, image_id: str,
                     image_prefix: Optional[str] = None, *args, **kwargs):
    prompts = make_prompt_by_categorical(
        PARTIAL_PREFIX, prompt1, Categorical1, df)
    return [{"prompt": prompt1, "color": color, "texture": texture,
             "image_id": generate_image(prompt, image_id, image_prefix)} for prompt, color, texture in prompts]


def generate_partial(
    prompt1: str,
    Categorical1: List,
    Numerical: List,
    df: pd.DataFrame,
    image_id: str,
    image_prefix: Optional[str] = None,
    *args,
    **kwargs,
):
    prompts = make_prompt_by_categorical(PARTIAL_PREFIX, prompt1, Categorical1, df)
    return [
        {
            "prompt": prompt1,
            "color": color,
            "texture": texture,
            "image_id": generate_image(prompt, image_id, image_prefix),
        }
        for prompt, color, texture in prompts
    ]


def generate_whole(
    prompt1: Union[str, List],
    Categorical1: List,
    Numerical: List,
    df: pd.DataFrame,
    image_id: str,
    image_prefix: Optional[str] = None,
    *args,
    **kwargs,
):
    if isinstance(prompt1, list):
        _image_id = []
        for prompt in prompt1:
            prompts = make_prompt_by_categorical(WHOLE_PREFIX, prompt, Categorical1, df)
            _image_id += [
                {
                    "prompt": prompt,
                    "color": color,
                    "texture": texture,
                    "image_id": generate_image(prompt, image_id, image_prefix),
                }
                for prompt, color, texture in prompts
            ]
        return _image_id
    prompts = make_prompt_by_categorical(WHOLE_PREFIX, prompt1, Categorical1, df)
    return [
        {
            "prompt": prompt1,
            "color": color,
            "texture": texture,
            "image_id": generate_image(prompt, image_id, image_prefix),
        }
        for prompt, color, texture in prompts
    ]


def generate_combination(
    image_id: List,
    prompt1: Union[str, List],
    prompt2: str,
    Categorical1: List,
    Categorical2: List,
    df: pd.DataFrame,
    Numerical: List,
    *args,
    **kwargs,
):
    main_images = [
        generate_whole(prompt1, Categorical1, Numerical, df, id_, "main_generated_")
        for id_ in image_id[:-1]
    ]
    sub_image = generate_partial(
        prompt2, Categorical2, Numerical, df, image_id[-1], "sub_generated_"
    )
    return main_images + sub_image


def generate_image(prompt: str, image_id: str, image_prefix: None = None):
    img = np.array(get_image_by_id(image_id))
    low_threshold = 100
    high_threshold = 200
    img = cv2.Canny(img, low_threshold, high_threshold)
    img = img[:, :, None]
    img = np.concatenate([img, img, img], axis=2)
    canny_image = Image.fromarray(img)

    out = PIPE_SDCC(prompt, num_inference_steps=20, image=canny_image).images[0]
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
        regenerate_prompt, image_id, "_".join(image_id.split("_")[:-1])
    )
    return {
        "image_id": image_id,
        "prompt": prompt,
        "texture": texture,
        "color": color,
    }


# image process


def numerical_partial(
    image: Image.Image,
    color: Optional[str],
    texture: Optional[str],
    color_column: Optional[str],
    texture_column: Optional[str],
    numerical: str,
    df: pd.DataFrame,
    process_type: str,
    image_pipe: List,
    *args,
    **kwargs,
):
    data = df[
        (color is None or df[color_column] == color)
        & (texture is None or df[texture_column] == texture)
    ]
    column_name = numerical["column"]
    column_value = numerical["value"]
    if column_name == "number":
        for number in data[column_value]:
            image_pipe.append(
                scale_image(process_to_radiation(image, number))
                if process_type
                else process_to_circle(image, number)
            )
    if column_name == "size":
        for idx, size in enumerate(data[column_value]):
            image_width, image_height = scale_size(
                df[column_value].max(), df[column_value].min(), size, image
            )
            if image_pipe:
                image_pipe[idx] = image_pipe[idx].resize((image_width, image_height))
            else:
                image_pipe.append(image.resize((image_width, image_height)))
    if column_name == "opacity":
        for idx, opacity in enumerate(data[column_value]):
            if image_pipe:
                image_pipe[idx] = set_alpha(
                    image_pipe[idx],
                    calculate_opacity(
                        df[column_value].max(), df[column_value].min(), opacity
                    ),
                )
            else:
                image_pipe.append(
                    set_alpha(
                        image,
                        calculate_opacity(
                            df[column_value].max(), df[column_value].min(), opacity
                        ),
                    )
                )


def numerical_whole(
    image: Image.Image,
    color: Optional[str],
    texture: Optional[str],
    color_column: Optional[str],
    texture_column: Optional[str],
    size_of_whole: int,
    numerical: str,
    df: pd.DataFrame,
    process_type: int,
    image_pipe: List,
    *args,
    **kwargs,
):
    data = df[
        (color is None or df[color_column] == color)
        & (texture is None or df[texture_column] == texture)
    ]
    column_name = numerical["column"]
    column_value = numerical["value"]
    if column_name == "number":
        for number in data[column_value]:
            image_pipe.append(
                process_to_transverse(image, size_of_whole, number)
                if process_type
                else process_to_vertical(image, size_of_whole, number)
            )
    if column_name == "size":
        for idx, size in enumerate(data[column_value]):
            image_width, image_height = scale_size(
                df[column_value].max(), df[column_value].min(), size, image
            )
            if image_pipe:
                image_pipe[idx] = image_pipe[idx].resize((image_width, image_height))
            else:
                image_pipe.append(image.resize((image_width, image_height)))
    if column_name == "opacity":
        for idx, opacity in enumerate(data[column_value]):
            if image_pipe:
                image_pipe[idx] = set_alpha(
                    image_pipe[idx],
                    calculate_opacity(
                        df[column_value].max(), df[column_value].min(), opacity
                    ),
                )
            else:
                image_pipe.append(
                    set_alpha(
                        image,
                        calculate_opacity(
                            df[column_value].max(), df[column_value].min(), opacity
                        ),
                    )
                )


def numerical_combination(
    image: Image.Image,
    color: Optional[str],
    texture: Optional[str],
    color_column: Optional[str],
    texture_column: Optional[str],
    df: pd.DataFrame,
    process_type: int,
    numerical: Dict,
    sub_image: Image.Image,
    image_pipe: List,
    numerical_number1: Dict,
    *args,
    **kwargs,
):
    try:
        data = df[
            (color is None or df[color_column] == color)
            & (texture is None or df[texture_column] == texture)
            & (
                sub_image.get("color") is None
                or df[sub_image["color"]] == sub_image["color_column"]
            )
            & (
                sub_image.get("texture") is None
                or df[sub_image["texture"]] == sub_image["texture_column"]
            )
        ]
    except KeyError:
        return
    if data.empty:
        return
    sub_image = get_image_by_id(sub_image["image_id"])
    column_name = numerical["column"]
    column_value = numerical["value"]
    if column_name == "number1" and process_type:
        for sub_of_number, main_number in zip(
            data[numerical_number1["value"]], data[column_value]
        ):  # TODO: wait fix
            main_image = (
                scale_image(process_to_radiation(image, main_number))
                if process_type
                else process_to_circle(image, main_number)
            )
            image_pipe.append(
                process_to_combination(main_image, sub_image, sub_of_number)
            )
    if column_name == "number" and process_type == 0:
        for number in data[column_value]:
            image_pipe.append(process_to_combination(image, sub_image, number))
    if column_name == "size":
        for idx, size in enumerate(data[column_value]):
            if image_pipe:
                image_pipe[idx] = image_pipe[idx].resize(
                    scale_size(
                        df[column_value].max(), df[column_value].min(), size, image
                    )
                )
            else:
                image_pipe.append(
                    process_to_combination(
                        resize_image_of_combination(image, size),
                        sub_image,
                        df["number"][idx],
                    )
                )
    if column_name == "opacity":
        for idx, opacity in enumerate(data[column_name]):
            if image_pipe:
                image_pipe[idx] = set_alpha(
                    image_pipe[idx],
                    calculate_opacity(
                        df[column_name].max(), df[column_name].min(), opacity
                    ),
                )
            else:
                image_pipe.append(
                    set_alpha(
                        image,
                        calculate_opacity(
                            df[column_name].max(), df[column_name].min(), opacity
                        ),
                    )
                )


PROCESS = {
    "partial": numerical_partial,
    "whole": numerical_whole,
    "combination": numerical_combination,
}


def scale_size(max_of_data, min_of_data, size, image) -> Tuple[int, int]:
    loss = max_of_data - min_of_data
    image_width = image.size[0] + (image.size[0] / loss) * (size - loss)
    image_height = image.size[1] + (image.size[1] / loss) * (size - loss)
    return int(image_width), int(image_height)


def calculate_opacity(max_of_opacity, min_of_opacity, opacity):
    return 25 + ((opacity - min_of_opacity) / (max_of_opacity - min_of_opacity)) * (
        100 - 25
    )


def process_image_by_numerical(data: Dict):
    data_title = data["data_title"]
    df = pd.read_csv(os.path.join(DATAPATH, data_title + ".csv"))
    images = data.get("images")
    Numerical = data.get("Numerical")
    design = data.get("design")
    sub_images = (
        [item for item in images if "sub_" in item["image_id"]]
        if design == "combination"
        else None
    )
    images = (
        [item for item in images if "main_" in item["image_id"]]
        if design == "combination"
        else images
    )

    result_images = []
    for item in images:
        image_id = item.get("image_id")
        color = item.get("color")
        texture = item.get("texture")
        color_column = item.get("color_column")
        texture_column = item.get("texture_column")

        image = get_image_by_id(image_id)
        image_pipe = []

        numericals = [numerical["column"] for numerical in Numerical]
        numerical_number1 = (
            Numerical[numericals.index("number1")] if "number1" in numericals else None
        )

        if sub_images:
            for sub_image in sub_images:
                for numerical in Numerical:
                    PROCESS[design](
                        image=image,
                        color=color,
                        texture=texture,
                        df=df,
                        color_column=color_column,
                        texture_column=texture_column,
                        numerical=numerical,
                        image_pipe=image_pipe,
                        sub_image=sub_image,
                        numerical_number1=numerical_number1,
                        **data,
                    )

        else:
            for numerical in Numerical:
                PROCESS[design](
                    image=image,
                    color=color,
                    texture=texture,
                    df=df,
                    color_column=color_column,
                    texture_column=texture_column,
                    numerical=numerical,
                    image_pipe=image_pipe,
                    sub_image=None,
                    numerical_number1=numerical_number1,
                    **data,
                )
        result_images += image_pipe
    return [save_image(img, "res_") for img in result_images]


def resize_image_of_combination(image, scale_factor):
    # 加载图像
    # 获取图像的尺寸
    original_width, original_height = image.size

    # 计算新的尺寸
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    # 缩小狗的图像
    resized_dog_image = image.resize((new_width, new_height), Image.ANTIALIAS)

    # 创建一个透明背景图像
    background = Image.new("RGBA", (original_width, original_height), (0, 0, 0, 0))

    # 计算粘贴的位置（使狗的图像在背景中居中）
    paste_x = (original_width - new_width) // 2
    paste_y = (original_height - new_height) // 2

    # 粘贴缩小的狗图像到背景中
    background.paste(resized_dog_image, (paste_x, paste_y), resized_dog_image)

    return background


def set_alpha(img, alpha_percentage) -> Image.Image:
    """
    调整图像的透明度。

    参数:
        img (Image.Image): 一个Pillow图像实例。
        alpha_percentage (float): 透明度百分比，范围在0到100之间。0表示完全透明，100表示不透明。

    返回:
        Image.Image: 调整透明度后的图像。
    """
    assert 0 <= alpha_percentage <= 100, "Alpha percentage should be between 0 and 100."

    # 如果原图不是RGBA模式，转化为RGBA模式
    img = img.convert("RGBA")

    # 获取图像的每个像素的RGBA值
    datas = img.getdata()

    new_data = []
    for item in datas:
        # 修改alpha值
        new_data.append(
            (item[0], item[1], item[2], int(item[3] * alpha_percentage / 100))
        )

    # 应用新的透明度数据
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
            draw.line([(start_x, start_y), (end_x, end_y)], fill=fill, width=width)


def make_grid(
    images: List[str],
    border_thickness: int,
    color_fill: str,
    dashed: bool,
    draw_lines: bool,
    background_type: str,
    background_color: str,
    *args,
    **kwargs,
) -> str:
    def center_crop(image, target_width, target_height):
        """裁剪图像使其大小为目标宽度和高度，同时保持中间的内容不变。"""
        center_x = image.width // 2
        center_y = image.height // 2

        left = center_x - target_width // 2
        top = center_y - target_height // 2
        right = center_x + target_width // 2
        bottom = center_y + target_height // 2

        return image.crop((left, top, right, bottom))

    flower_images = [get_image_by_id(image_id) for image_id in images]
    N = math.ceil(math.sqrt(len(flower_images)))

    output_width = N * 400
    output_height = N * 400

    if background_type == "transparent":
        output_image = Image.new(
            "RGBA", (output_width, output_height), (255, 255, 255, 0)
        )
    elif background_type == "color":
        output_image = Image.new(
            "RGBA", (output_width, output_height), background_color
        )

    # 将每张花的图片粘贴到网格的背景图像上
    for idx, flower_image in enumerate(flower_images):
        row = idx // N
        col = idx % N

        cropped_flower = center_crop(flower_image, 400, 400)
        x_offset = col * 400
        y_offset = row * 400
        output_image.paste(cropped_flower, (x_offset, y_offset), cropped_flower)

    # 在网格上添加黑色边框
    draw = ImageDraw.Draw(output_image)

    # 绘制垂直线
    if draw_lines:
        for i in range(1, N**2):
            x = i * 400
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
        for i in range(1, N**2):
            y = i * 400
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

    return save_image(output_image.resize((512, 512)), "placement_")


def make_struct(
    images: List[str],
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
    *args,
    **kwargs,
) -> str:
    df = pd.read_csv(os.path.join(DATAPATH, data_title + ".csv"))
    fig, ax = plt.subplots(figsize=(5.12, 5.12), dpi=500)

    images = [get_image_by_id(image_id) for image_id in images]

    # 设置背景
    if background_type == "transparent":
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
    elif background_type == "color":
        fig.patch.set_facecolor(canvas_color)
        ax.set_facecolor(background_color)

    # 在指定的year和score位置上放置图片
    for x_data, y_data, img in zip(df[column1], df[column2], images):
        imagebox = OffsetImage(img, zoom=0.15)  # 这里的zoom可以调整
        ab = AnnotationBbox(imagebox, (x_data, y_data), frameon=False)
        ax.add_artist(ab)

    # 设定x轴和y轴的标签以及其样式
    ax.set_xlabel(column1, fontsize=text_size, color=text_color, weight="bold")
    ax.set_ylabel(column2, fontsize=text_size, color=text_color, weight="bold")
    ax.scatter(df[column1], df[column2], alpha=0)  # 透明的散点，只为确定轴的范围

    # 设置轴的颜色
    ax.spines["bottom"].set_color(text_color)
    ax.spines["top"].set_color(text_color)
    ax.spines["left"].set_color(text_color)
    ax.spines["right"].set_color(text_color)
    ax.tick_params(axis="both", colors=text_color, which="major", labelsize=text_size)

    # 显示网格
    if show_grid:
        ax.grid(True, which="both", linestyle="--", linewidth=1.5, color=grid_color)
        ax.set_axisbelow(True)

    # 保存图形
    plt.tight_layout()
    filename = f"placement_{int(datetime.datetime.now().timestamp())}"
    plt.savefig(
        os.path.join(IMAGE_RESOURCE_PATH, filename + ".png"),
        dpi=500,
        bbox_inches="tight",
        pad_inches=0,
        transparent=True if background_type == "transparent" else False,
    )
    return filename


def make_geo(
    images: List[str],
    data_title: str,
    column1: str,
    column2: str,
    fill_color: str,
    continent_color: str,
    countries_color: str,
    linestyle: str,
    coastlines: str,
    lake_color: str,
    *args,
    **kwargs,
) -> str:
    data = pd.read_csv(os.path.join(DATAPATH, data_title + ".csv"))
    images = [get_image_by_id(image_id) for image_id in images]

    # 计算涵盖所有城市的经纬度范围
    lats = data[column1]
    lons = data[column2]

    # 初始化地图
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

    # 设置地图的颜色和边框
    m.drawmapboundary(fill_color=fill_color, linewidth=0)
    m.fillcontinents(color=continent_color, lake_color=lake_color)
    m.drawcountries(linewidth=2, linestyle=linestyle, color=countries_color)
    m.drawcoastlines(linewidth=0.5, color=coastlines)

    # 在地图上添加城市的图片
    for lat, lon, image in zip(lats, lons, images):
        x, y = m(lon, lat)
        size_factor = 0.1  # 使图片大小与年份相关
        img = OffsetImage(image, zoom=size_factor)
        ax.add_artist(AnnotationBbox(img, (x, y), frameon=False))

    filename = f"placement_{int(datetime.datetime.now().timestamp())}"
    plt.savefig(os.path.join(IMAGE_RESOURCE_PATH, filename + ".png"), dpi=500)
    return filename


def make_drawer_by_height(
    images: List[str],
    draw_lines: bool,
    line_width: int,
    line_color: str,
    # dashed: bool,
    background_color: Optional[str] = None,
    *args,
    **kwargs,
) -> str:
    images = [get_image_by_id(image_id) for image_id in images]
    total_height = sum(img.height for img in images)

    result_image = Image.new("RGBA", (total_height, total_height), background_color)
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
    return save_image(Image.fromqimage(result_image.toqimage()), "placement_")


def make_drawer_by_width(
    images: List[str],
    draw_lines: bool,
    line_width: int,
    line_color: str,
    background_color: Optional[str] = None,
    *args,
    **kwargs,
):
    images = [get_image_by_id(image_id) for image_id in images]
    total_width = sum(img.width for img in images)

    result_image = Image.new("RGBA", (total_width, total_width), background_color)
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
