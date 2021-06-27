import io

import dash
import dash_core_components as dcc
import dash_html_components as html
import faiss
import numpy as np
import torch
from tqdm.contrib.concurrent import thread_map
from text_autoencoders.model import DAE
from text_autoencoders.vocab import Vocab
from text_autoencoders.batchify import get_batch
from yelp_faiss import init_model


# Init Dash
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, assets_folder=".")
app.title = "Yelp Sentiment Similarity Search"

# Init Encoder
device = torch.device("cuda:0")
vocab = Vocab("text_autoencoders/checkpoints/yelp/daae/vocab.txt")
model = init_model("text_autoencoders/checkpoints/yelp/daae/model.pt", vocab, device)
print("Model Initialized")

# if not using a GPU, just call FAISS_INDEX = faiss.read_index
FAISS_INDEX = faiss.index_cpu_to_gpu(
    faiss.StandardGpuResources(), 0, faiss.read_index("yelp_flat.faiss")
)
print("FAISS Loaded")

# load path listing faiss was trained on.
SENTS = open("text_autoencoders/data/yelp/sentiment/1000.pos").readlines()

app.layout = html.Div(
    children=[
        html.H1(children="Yelp Similarity Search"),
        dcc.Input(id="upload-data", type="text"),
        html.Div(
            dcc.Slider(
                id="num-like",
                min=0,
                max=30,
                step=2,
                value=10,
            ),
            style={"width": "50%"},
        ),
        html.Div(id="slider-output-container"),
        html.Button("Search", id="search-button"),
        html.Div(id="output-container"),
    ]
)


@app.callback(
    dash.dependencies.Output("slider-output-container", "children"),
    [dash.dependencies.Input("num-like", "value")],
)
def update_output(value):
    return f"Num to Search: {value}"


@app.callback(
    dash.dependencies.Output("output-container", "children"),
    [dash.dependencies.Input("search-button", "n_clicks")],
    [
        dash.dependencies.State("upload-data", "value"),
        dash.dependencies.State("num-like", "value"),
    ],
)
def search_data(n_clicks: int, contents: str, n: int) -> html.Div:
    if n_clicks is None:
        return

    # Acquire vector
    batch_of_one, _ = get_batch(
        [contents],
        vocab,
        device,
    )
    with torch.no_grad():
        vector, _ = model.encode(batch_of_one.to(device))
    flat_vector = vector.flatten().cpu()

    # search the index
    d, i = FAISS_INDEX.search(np.expand_dims(flat_vector.numpy(), 0), n)
    nearby_sents = [html.Div(SENTS[n][:-3]) for n in i.squeeze()]

    # load the results!
    return html.Div(nearby_sents)


if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0")
