import pickle as pkl
import argparse
import os 
import numpy as np
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    args = parser.parse_args()

    with open(os.path.join(args.path, "result.pkl"), "rb") as f:
        data = pkl.load(f)

    labels = data['label']
    embeddings = data['embeddings']

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(np.concatenate(embeddings, axis=0))

    plt.figure(figsize=(16,10))
    plt.title("t-SNE visualization of the embeddings")
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=np.concatenate(labels), cmap='tab20')
    plt.colorbar()
    plt.savefig(os.path.join(args.path, "tsne.png"))
    