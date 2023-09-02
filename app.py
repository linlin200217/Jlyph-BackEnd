from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
from utils import COLOR, TEXTURE, data_process, make_glyph, IMAGE_RESOURCE_PATH, DATAPATH, generate_glyph, \
    regenerate_by_prompt

app = Flask(__name__)


@app.route("/upload")
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


@app.route("/pregenerate")
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
    image_id = make_glyph(request.form)
    return jsonify({"status": "success", "image_id": image_id})


@app.route("/generate")
async def generate():
    """
    POST
    {
        design: "partial" | "whole" | "combination",
        Categorical1: 
            [
                {column: "Flower Country", value: color | texture},
                {column: "Flower Type", value: color | texture},
            ],
        Categorical2:
            [
                {column: column1, value: color | texture},
                {column: column2, value: color | texture}
            ], | null,
        Numerical: ["number","size","opacity"],  # todo
        Ordinal: ["size","opacity"],
        prompt1: "",
        prompt2: "" | null,
        guide1:  0 | 1 | 2,
        guide2:  0 | 1 | 2 | null,
        image_id: [str],
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
    image_id = generate_glyph(request.form)
    return jsonify({"status": "success", "image_id": image_id})


@app.route("/regenerate")
def regenerate():
    """
    POST
    only call after generate
    {
        image_id: str,
        design: "partial" | "whole" | "combination",
        prompt: str,
        guide: 0 | 1 | 2,
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
    return regenerate_by_prompt(**request.form)


@app.route('/color')
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
    exist_color = request.form["exist_color"]
    return jsonify({"color": set(COLOR) - set(exist_color)})


@app.route('/texture')
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
    exist_texture = request.form["exist_texture"]
    return jsonify({"color": set(TEXTURE) - set(exist_texture)})


@app.route("/process")
def image_process():
    """
    {
        design: "partial" | "whole" | "combination",
        images: [
            {image_id: str, color: str, texture: str, shape?: str}
            ...
        ],
        Numerical: ["number","size","opacity"...]， # ["number" of sub, "number1" of main] if type of combbination
        process_type: 0 | 1 | 2<note: only by combbination>,
        size_of_whole?: int,
        data_title: str
    }
    """
    pass


@app.route("/image/<image_id>")
def get_image(image_id):
    """
    GET: HOST/image/<image_id> 
    return:
        image
    """
    return send_file(os.path.join(IMAGE_RESOURCE_PATH, image_id + ".png"))


if __name__ == "__main__":
    app.run(port=8000)
