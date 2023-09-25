from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
from utils import COLOR, TEXTURE, data_process, make_glyph, IMAGE_RESOURCE_PATH, DATAPATH, generate_glyph, \
    regenerate_by_prompt, process_image_by_numerical, placement

app = Flask(__name__)


@app.route("/upload", methods=["POST"])
async def load_data():
    """
    POST FILE
    return json data {
        data_title: string,
        Categorical: [],
        Numerical: [],
        wordcloud: {}
    }
    """
    if request.method == 'POST':
        f = request.files['file']
        file_path = os.path.join(DATAPATH, secure_filename(f.filename))
        f.save(file_path)
        struct = data_process(file_path)
        return jsonify(struct)


@app.route("/pregenerate", methods=["POST"])
async def pregenerate():
    """
    POST
    {
        design: "partial" | "whole" | "combination",
        prompt1: [subprompt1, subprompt2...] | prompt,
        prompt2: "" | null,
        guide1:  0 | 1 | 2,
        guide2:  0 | 1 | 2 | null,
    }

    return:
        {
            "status": "success", 
            "image": [
                {prompt: promt1 | subprompt1, image_id: <image_id of str>},
                {prompt: promt2 | subprompt2, image_id: <image_id of str>},
            ]
        }
    """
    image_id = make_glyph(request.json)
    return jsonify({"status": "success", "image_id": image_id})


@app.route("/generate", methods=["POST"])
async def generate():
    """
    POST
    {
        design: "partial" | "whole" | "combination",
        Categorical1: 
            [
                {column: "Flower Country", value: "color"},
                {column: "Flower Type", value: "texture"},
                {column: "Any", value: "shape"},
            ],
        Categorical2:
            [
                {column: column1, value: color | texture},
                {column: column2, value: color | texture}
            ], | null,
        Ordinal: ["size","opacity"],
        prompt1: [subprompt1, subprompt2...] | prompt,
        prompt2: "" | null,
        guide1:  0 | 1 | 2,
        guide2:  0 | 1 | 2 | null,
        image_id: [{
            "image_id": str,
            "prompt": str,
            "shape"?: str
        }...] | str, # whole or combination | partial
        data_title: str
    }
    return:
    {
            "status": "success",
            "image": [
                {prompt: promt1 | subprompt1, texture?: str, color?: str, image_id: <image_id of str>},
                {prompt: promt2 | subprompt2, texture?: str, color?: str, image_id: <image_id of str>},
            ]
    }

    """
    image_id = generate_glyph(request.json)
    return jsonify({"status": "success", "image_id": image_id})


@app.route("/regenerate", methods=["POST"])
def regenerate():
    """
    POST
    only call after generate
    {
        image_id: str,
        design: "partial" | "whole" | "combination",
        prompt: str,
        color?: str,
        texture?: str
    }
    return {
        image_id: str,
        prompt: str,
        texture?: str,
        color?: str
    }
    """
    return regenerate_by_prompt(**request.json)


@app.route('/color', methods=["POST"])
def get_color():
    """
    POST
    {
        exist_color: list[str]
    }
    return
    {
        color: list[str]
    }
    """
    exist_color = request.json["exist_color"]
    return jsonify({"color": set(COLOR) - set(exist_color)})


@app.route('/texture', methods=["POST"])
def get_texture():
    """
    POST
    {
        exist_texture: list[str]
    }
    return
    {
        texture: list[str]
    }
    """
    exist_texture = request.json["exist_texture"]
    return jsonify({"color": set(TEXTURE) - set(exist_texture)})


@app.route("/process", methods=["POST"])
def image_process():
    """
    {
        design: "partial" | "whole" | "combination",
        images: [
            {image_id: str, color: str, color_colunm: str, texture: str,texture_colunm: str, shape: str, shape_column: str}
        ],
        Numerical: [
            {
            column: "number1",
            value: "xxxxxXXxx"
            },
            {
            column: "number",
            value: "xxxxxXXxx"
            },
            {
            column: "size",
            value: "xxxxxXXxx"
            },
            {
            column: "opacity",
            value: "xxxxxXXxx"
            }], # ["number" of sub, "number1" of main] if type of combbination, number1 must in first
        process_type: 0 | 1 | 2<note: only by combination>, # 0: whole+partial, if type is 1 or 2, must add number1 of Numerical
        size_of_whole?: int,
        data_title: str
    }

    retrun {
        status: str,
        images: list[str]
    }

    局部：
    categorical：color，texture
    Numerical：Size，number，opacity

    整体：
    categorical：color，texture，shape
    numerical：SIze，number，opacity

    组合：
    categorical：color_main, color_auxiliary, texture_main, texture_auxiliary, shape_main
    Numerical: Number_main, Number_auxiliary, Size, Opacity
    注：只有在没有选择shape_main的前提下才可以选择Number_main，换句话说，如果选择了shape_main就没有number_main的选择。如果选择了number_main，就没有shape_main的选择。

    """
    return jsonify({"status": "success",
                    "images": [
                        {"data_index": column_index,
                         "image_id": image_id} for column_index, image_id in process_image_by_numerical(request.json)
                    ]})


@app.route("/placement", methods=["POST"])
def placement_image():
    """
    {
        design: "partial" | "whole" | "combination",
        drawer_by: "width" | "height",
        images: list[{
            "data_index": int, "image_id": str,
            ...
        }],
        method: "grid" | "struct" | "geo",
        data_title: str,
        border_thickness: int,    # grid
        color_fill: str,          # grid
        dashed: bool,             # grid
        draw_lines: bool,         # grid
        column1: str,             # struct, lat of geo
        column2: str,             # struct, lon of geo
        canvas_color: str,        # struct
        text_size: int,           # struct
        text_color: str,          # struct
        grid_color: str,          # struct
        show_grid: bool,           # struct
        fill_color: str,          # geo
        continent_color: str,     # geo
        countries_color: str,     # geo
        linestyle: str,           # geo
        coastlines: str,          # geo
        lake_color: str,          # geo
        background_type: str of ['transparent', 'color']  # grid, struct
        background_color: str,     # struct, grid
    }
    return
    {
        status: "success",
        image_id: str
    }
    """
    return jsonify({"status": "success", "image_id": placement(request.json)})


@app.route("/image/<image_id>")
def get_image(image_id):
    """
    GET: HOST/image/<image_id> 
    return:
        image
    """
    return send_file(os.path.join(IMAGE_RESOURCE_PATH, image_id + ".png"))


if __name__ == "__main__":
    app.run(port=8000, debug=True)
