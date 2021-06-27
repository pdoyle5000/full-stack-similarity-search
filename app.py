import base64
import io
import pickle

import dash
import dash_core_components as dcc
import dash_html_components as html
import faiss
import numpy as np
import torch
from PIL import Image
from tqdm.contrib.concurrent import thread_map

from autoencoder import ConvEncoder

# Init Dash
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Similarity Search"

# Init Encoder
encoder = ConvEncoder().cpu()
encoder.load_state_dict(
    torch.load("models/encoderfinal_nonorm_two.pth", map_location=torch.device("cpu"))
)
encoder.eval()


# if not using a GPU, just call FAISS_INDEX = faiss.read_index
FAISS_INDEX = faiss.index_cpu_to_gpu(
    faiss.StandardGpuResources(), 0, faiss.read_index("ivf_flat.faiss")
)

# load path listing faiss was trained on.
PATHS = pickle.load(open("path_index.pkl", "rb"))

app.layout = html.Div(
    children=[
        html.H1(children="Similarity Search"),
        dcc.Upload(
            id="upload-data",
            children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
            style={
                "width": "50%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            multiple=False,
        ),
        html.Div(id="image-preview"),
        html.Div(
            dcc.Slider(
                id="num-like",
                min=0,
                max=512,
                step=32,
                value=64,
            ),
            style={"width": "50%"},
        ),
        html.Div(id="slider-output-container"),
        html.Button("Search", id="search-button"),
        html.Div(id="image-container"),
    ]
)


@app.callback(
    dash.dependencies.Output("slider-output-container", "children"),
    [dash.dependencies.Input("num-like", "value")],
)
def update_output(value):
    return f"Num to Search: {value}"


@app.callback(
    dash.dependencies.Output("image-preview", "children"),
    [dash.dependencies.Input("upload-data", "contents")],
)
def preview_image(contents):
    if contents is None:
        return
    return html.Img(src=contents)


@app.callback(
    dash.dependencies.Output("image-container", "children"),
    [dash.dependencies.Input("search-button", "n_clicks")],
    [
        dash.dependencies.State("upload-data", "contents"),
        dash.dependencies.State("num-like", "value"),
    ],
)
def search_data(n_clicks: int, contents: str, n: int) -> html.Div:
    if n_clicks is None:
        return

    # Acquire vector
    decoded = base64.b64decode(contents.split(",")[1])
    image = np.array(Image.open(io.BytesIO(decoded))) / 255
    image = np.transpose(image.astype(float), (2, 0, 1))
    tensor = torch.from_numpy(image).float()
    with torch.no_grad():
        vector = encoder(tensor.unsqueeze(0))
    flat_vector = vector.flatten()

    # search the index
    d, i = FAISS_INDEX.search(np.expand_dims(flat_vector.numpy(), 0), n)
    nearby_paths = [PATHS[n] for n in i.squeeze()]

    # load the images!
    child = thread_map(render_image, nearby_paths)
    return html.Div(child)


def render_image(path: str) -> html.Img:
    src = base64.b64encode(open(path, "rb").read())  # noqa: F841
    return html.Img(src=f"data:image/png;base64,{src.decode()}")


if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0")
