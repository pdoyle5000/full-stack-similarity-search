import pickle
from typing import List, Tuple

import faiss  # type: ignore
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm  # type: ignore

from text_autoencoders.vocab import Vocab
from text_autoencoders.batchify import get_batches
from text_autoencoders.model import DAE


def load_sentiment(path: str) -> List[List[str]]:
    return [line.split() for line in open(path)]


def init_model(path: str, vocab: Vocab, device=torch.device("cuda:0")) -> DAE:
    checkpoint = torch.load(path)
    model = DAE(vocab, checkpoint["args"]).to(device)
    del checkpoint["model"]["D.0.weight"]
    del checkpoint["model"]["D.2.weight"]
    del checkpoint["model"]["D.0.bias"]
    del checkpoint["model"]["D.2.bias"]
    model.load_state_dict(checkpoint["model"])
    model.flatten()
    model.eval()
    return model


def generate_vectors(model: DAE, batches: List[List[str]]) -> torch.Tensor:
    vectors = torch.Tensor()
    with torch.no_grad():
        for inputs, _ in tqdm(batches, total=len(batches), desc="Generating Vectors"):
            vector, _ = model.encode(inputs)
            vectors = torch.cat((vectors, vector.flatten(start_dim=1).cpu()))
    return vectors


if __name__ == "__main__":

    # Stage data.
    path = "text_autoencoders/data/yelp/sentiment/1000.pos"
    vocab = Vocab("text_autoencoders/checkpoints/yelp/daae/vocab.txt")
    device = torch.device("cuda:0")
    # batches, _ = get_batches(open(path).readlines(), vocab, 64, device)
    batches, _ = get_batches(load_sentiment(path), vocab, 64, device)

    # Init Model.
    checkpoint_path = "text_autoencoders/checkpoints/yelp/daae/model.pt"
    model = init_model(checkpoint_path, vocab)

    vectors = generate_vectors(model, batches)

    # Load Faiss
    res = faiss.StandardGpuResources()
    index = faiss.IndexFlatL2(128)
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
    gpu_index_flat.train(vectors.numpy())
    gpu_index_flat.add(vectors.numpy())

    # Example Query
    print(vectors.numpy()[5].shape)
    d, i = gpu_index_flat.search(np.expand_dims(vectors.numpy()[5], 0), 5)

    with open(path) as f:
        lines = f.readlines()
        for n in i.squeeze():
            print(f"{n}: {lines[n][:-3]}")

    # Save Faiss object for dash app loading
    faiss.write_index(faiss.index_gpu_to_cpu(gpu_index_flat), "yelp_flat.faiss")
