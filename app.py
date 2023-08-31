import os
import sqlite3
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from utils import data_process, make_glyph, IMAGE_RESOURCE_PATH, DATAPATH


app = Flask(__name__)

def get_db_connection():
    conn = sqlite3.connect('./database/flower.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.route("/")
def index():
    return "Flask loaded"

@app.route("/upload")
async def load_data():
    """
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
    {
        design: "partial" | "whole" | "combination",
        prompt1: "",
        prompt2: "" | null,
        guide1:  0 | 1 | 2,
        guide2:  0 | 1 | 2 | null,
    }

    return:
        {
            "status": "success", 
            "image_id": image_id
        }
    """
    image_id = make_glyph(request.form)
    return jsonify({"status": "success", "image_id": image_id})


@app.route("/generate")
async def generate():
    """
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
        image_id: str,
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
    return send_file(os.path.join(IMAGE_RESOURCE_PATH, image_id+".png"))


if __name__=="__main__":
    app.run(port=8000)
