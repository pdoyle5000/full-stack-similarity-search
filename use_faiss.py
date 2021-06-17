import pickle
from typing import List, Tuple

import faiss  # type: ignore
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm  # type: ignore

from autoencoder import ConvEncoder
from autoencoder.dataset import CifarDataset


def generate_vectors(
    encoder: ConvEncoder, dataloader: DataLoader
) -> Tuple[torch.Tensor, List[str]]:
    # Create vectors
    vectors = torch.Tensor()
    paths = []
    with torch.no_grad():
        for (inputs, _, path) in tqdm(
            dataloader, total=len(dataloader), desc="Geneating Vectors"
        ):
            inputs = inputs.float().to(device)
            reconstructed_vectors = encoder(inputs).flatten(start_dim=1)
            vectors = torch.cat((vectors, reconstructed_vectors.cpu()))
            paths.extend(path)
    return vectors, paths


if __name__ == "__main__":

    # Stage data.
    dataset = CifarDataset("train")
    dataloader = DataLoader(dataset, batch_size=2048, shuffle=False)
    device = torch.device("cuda:0")

    # Init Models.
    encoder_path = "models/encoderfinal_nonorm_two.pth"
    encoder = ConvEncoder().to(device)
    encoder.load_state_dict(torch.load(encoder_path))
    encoder.eval()

    # Create hash tables
    vectors, paths = generate_vectors(encoder, dataloader)

    # Load Faiss
    res = faiss.StandardGpuResources()
    quantizer = faiss.IndexFlatL2(512)
    index_ivf = faiss.IndexIVFFlat(quantizer, 512, 10, faiss.METRIC_L2)
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_ivf)
    gpu_index_flat.train(vectors.numpy())
    gpu_index_flat.add(vectors.numpy())

    # Example Query
    d, i = gpu_index_flat.search(np.expand_dims(vectors.numpy()[5], 0), 5)

    # print(paths)
    for n in i.squeeze():
        print(n)
        print(paths[int(n)])

    # Save Faiss object for dash app loading
    faiss.write_index(faiss.index_gpu_to_cpu(gpu_index_flat), "ivf_flat.faiss")
    with open("path_index.pkl", "wb") as f:
        pickle.dump(paths, f)
