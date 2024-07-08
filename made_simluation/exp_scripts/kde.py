import os
import sys
import numpy as np
import pickle
import scipy
from scipy.stats import multivariate_normal
import sklearn
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

def load_data():
    x_train = np.load("../experiments/kde/no-attack/train_emb.npz")['train']
    with open("../experiments/tsne/result.pkl", "rb") as f:
        data = pickle.load(f)
    x_test = np.concatenate([e for e in data['embeddings'] if len(e) == 12])
    label = np.concatenate([l for l in data['label'] if len(l) == 12])
    
    print(f"Dimension: {x_train.shape[1]}")
    print(f"Training samples: {x_train.shape[0]}")
    print(f"Test samples: {x_test.shape[0]}, {np.sum(label == 0)} in-distribution, {np.sum(label == 1)} out-of-distribution")
    return x_train[:, :], x_test[:, :], label

def load_data_v2(dim):
    with open(f"../experiments/residual_ae_v3/{dim}/no-attack/result.pkl", 'rb') as f:
        data = pickle.load(f)
    x_train = np.concatenate([e for e in data['embeddings'] if len(e) != 12])

    with open(f"../experiments/residual_ae_v3/{dim}/shift/N01_E1e-1_S10_sep/result.pkl", "rb") as f:
        data = pickle.load(f)
    x_test = np.concatenate([e for e in data['embeddings'] if len(e) == 12])
    label = np.concatenate([l for l in data['label'] if len(l) == 12])
    
    print(f"Dimension: {x_train.shape[1]}")
    print(f"Training samples: {x_train.shape[0]}")
    print(f"Test samples: {x_test.shape[0]}, {np.sum(label == 0)} in-distribution, {np.sum(label == 1)} out-of-distribution")
    return x_train[:, :], x_test[:, :], label
# # Generate random data
# d = 2
# n_train = 100
# n_out = 100
# n_in = 100
# # Training samples (Gaussian) (n_train, d)
# mean = np.random.uniform(low=-1, high=1, size=(d,))
# cov = np.random.uniform(low=0, high=1, size=(d, d))
# cov = np.diag(np.diag(cov))
# x_train = np.random.multivariate_normal(mean=mean, cov=cov, size=(n_train,))
# # # Visualization
# # plt.scatter(x_train[:, 0], x_train[:, 1])
# # plt.show()
# # Test samples out-of-distribution (Uniform) (n_test, d)
# x_out = np.random.uniform(low=-1, high=1, size=(n_out, d))
# # Test samples in-distribution (Gaussian) (n_test, d)
# x_in = np.random.multivariate_normal(mean=mean, cov=cov, size=(n_in,))


# # Compute KDE for a test sample
def kde(x, x_train, sigma=1.):
    d = x_train.shape[1]
    cov = np.zeros((d, d), int)
    np.fill_diagonal(cov, pow(sigma, 2))
    log_density = 0
    for i in tqdm(range(x_train.shape[0])):
        log_density += np.log(multivariate_normal.pdf(x, mean=x_train[i, :], cov=cov))

    return log_density


# log_density_out = []
# for i in range(n_out):
#     log_density_out.append(kde(x_out[i, :], x_train))
# log_density_in = []
# for i in range(n_in):
#     log_density_in.append(kde(x_in[i, :], x_train))

# print(np.mean(log_density_out))
# print(np.mean(log_density_in))

def plot_recon_hist():
    for D in [64, 128, 256]:
        with open(f"../experiments/residual_ae_v3/{D}/shift/N01_E1e-1_S10_sep/result.pkl", "rb") as f:
            data = pickle.load(f)
        score = np.concatenate([s for s in data['score'] if len(s) == 12])
        label = np.concatenate([l for l in data['label'] if len(l) == 12])

        plt.figure(figsize=(5,5))
        plt.hist(score[label == 0], bins=100, alpha=0.5, label='normal')
        plt.hist(score[label == 1], bins=100, alpha=0.5, label='attacker')
        # plt.title(title)
        plt.legend()
        plt.savefig(f"D{D}/hist.png")
        plt.close()
    
# # Comment: in-distribution samples should have larger (log) density than out-of-distribution samples
if __name__ == "__main__":
    # plot_recon_hist()
    # exit()
    D = 64
    if not os.path.exists(f"D{D}"):
        os.makedirs(f"D{D}")
    x_train, x_test, label = load_data_v2(D)
    # log_density = np.array([kde(x, x_train) for x in tqdm(x_test)])
    # print(f"Mean log density of in-distribution samples: {np.mean(log_density[label == 0])}")
    # print(f"Mean log density of out-of-distribution samples: {np.mean(log_density[label == 1])}")
    
    d = 64
    # PCA
    if d < D:
        pca = PCA(n_components=d)
        pca.fit(x_train)
        x_train = pca.transform(x_train)
        x_test = pca.transform(x_test)

    # KDE with sklearn
    bw = 0.05
    kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(x_train)
    log_density = kde.score_samples(x_test)
    print(f"Mean log density of in-distribution samples: {np.mean(log_density[label == 0])}")
    print(f"Mean log density of out-of-distribution samples: {np.mean(log_density[label == 1])}")

    plt.figure(figsize=(15,5))
    plt.title(f"KDE distribution d={d}, 2800 training samples")
    plt.hist(log_density[label == 0], bins=200, alpha=0.5, label="in-distribution")
    plt.hist(log_density[label == 1], bins=200, alpha=0.5, label="out-of-distribution")
    plt.legend()
    plt.savefig(f"D{D}/kde_{d}_bw{bw}.png")
    plt.close()


    # GMM
    gmm = GaussianMixture(n_components=5, covariance_type='full').fit(x_train)
    log_density = gmm.score_samples(x_test)
    print(f"Mean log density of in-distribution samples: {np.mean(log_density[label == 0])}")
    print(f"Mean log density of out-of-distribution samples: {np.mean(log_density[label == 1])}")

    plt.figure(figsize=(15,5))
    plt.title(f"GMM distribution d={d}, 2800 training samples")
    plt.hist(log_density[label == 0], bins=200, alpha=0.5, label="in-distribution")
    plt.hist(log_density[label == 1], bins=200, alpha=0.5, label="out-of-distribution")
    plt.legend()
    plt.savefig(f"D{D}/gmm_{d}_ncomp{5}.png")
    plt.close()
    