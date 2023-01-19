import sys, os
import math
from abc import ABC
from abc import abstractmethod

import numpy as np 
import scipy as sp
import pandas as pd
from PIL import Image
from skimage.feature import local_binary_pattern as lbp_scikit




# BUILDING BLOCKS

def load_image(path, h=128, w=128, numpy=True):
    img = Image.open(path)
    #to_tensor = 
    img = img.convert("L")
    img = img.resize((h,w))
    if numpy:
        img = np.array(img)
    return img

def right_rotate_int(a, shift, num_bits):
    return (a >> shift) | (a << (num_bits - shift)) & ((1 << num_bits) - 1)

def get_nbh_lookup(P, R):
    nbh_lookup = []

    for p in range(P):
        dy = round(R * math.sin(2*math.pi * p / P))
        dx = round(R * math.cos(2*math.pi * p / P))
        nbh_lookup.append((dy, dx))

    return nbh_lookup







# FEATURE EXTRACTOR CLASSES


class Extractor(ABC):

    @abstractmethod
    def __call__(self, img):
        """Extracts features for every pixel in given image.

        Args:
            img: An image as a (h, w) ndarray.

        Returns:
            An ndarray of size (h, w): a "feature image" where every pixel
                represents the pattern at that pixel in the original image.
        """


class PixelValues(Extractor):
    """Returns the image where features are pixel values: i.e. the original img."""

    def __init__(self, float=True):
        self.float = float
        self.d = 256

    def __call__(self, img):
        return img


def get_uniform_lbp(img, row, col, nbh_lookup, rot_inv=False):
    
    pattern = 0
    P = len(nbh_lookup)
    for p in range(P):
        dy, dx = nbh_lookup[p]
        value = (img[row+dy, col+dx] > img[row, col])
        pattern = pattern * 2 + value

    if rot_inv:
        shifts = [right_rotate_int(pattern, p, P) for p in range(P)]
        pattern = min(shifts)

    return int(pattern)

class BinaryPattern(Extractor):
    """Returns an image of uniform binary patterns for every pixel."""

    def __init__(self, P=8, R=1, rot_inv=False, pad="default"):
        self.P = P 
        self.R = R 
        self.rot_inv = rot_inv
        self.pad = R if pad == "default" else pad
        self.d = 2**P
    
    def __call__(self, img):
        
        h, w = img.shape
        pad = self.pad
        lbp_img = np.ndarray((h-2*pad, w-2*pad)).astype(int)

        nbh_lookup = get_nbh_lookup(self.P, self.R)
        
        for row in range(pad, h - pad):
            for col in range(pad, w - pad):
                lbp_img[row-pad, col-pad] = get_uniform_lbp(img, row, col,
                                                    nbh_lookup, rot_inv=self.rot_inv)

        return lbp_img


def get_lbp_img_fast(img, P, R):

    h, w = img.shape
    pattern_block = np.ndarray((P, h-2*R, w-2*R))
    shifts = get_nbh_lookup(P, R)

    for i in range(P):
        dy, dx = shifts[i]
        pattern_block[i, :, :] = img[R-dy: h-R-dy, R-dx: w-R-dx]
    
    tiled_img = np.repeat(img[np.newaxis, :, :], P, axis=0)
    pattern_block = -pattern_block + img[R: h-R, R: w-R]
    pattern_block = -np.sign(pattern_block).astype(int)
    pattern_block[pattern_block <= 0] = 0

    for p in range(P):
        exp = P-p-1
        pattern_block[p, :, :] *= 2**exp

    lbp_img = np.sum(pattern_block, axis=0, keepdims=False)
    return lbp_img

class FastBinaryPattern(Extractor):
    def __init__(self, P=8, R=1):
        self.P = P 
        self.R = R 
        self.d = 2**P

    def __call__(self, img):

        lbp_img = get_lbp_img_fast(img, self.P, self.R)
        return lbp_img


class ScikitBinaryPattern(Extractor):

    def __init__(self, P=8, R=1):
        self.P = P 
        self.R = R 
        self.d = 2**P

    def __call__(self, img):

        lbp_img = lbp_scikit(img, self.P, self.R, "uniform")
        return lbp_img





def get_histogram(ft_img, d, row, col, cell_size):
    
    cell = ft_img[row:row+cell_size, col:col+cell_size]
    hist = np.ndarray(d)

    unique, counts = np.unique(ft_img, return_counts=True)
    unique = unique.astype(int)
    hist[unique] = counts

    return hist

def process_hists(ft_img, d, cell_size=16, mode="mean"):

    hist_list = []
    rows = int( ft_img.shape[0] / cell_size )
    cols = int( ft_img.shape[1] / cell_size )

    for i in range(rows):
        cell_row = i*cell_size

        for j in range(cols):
            cell_col = j*cell_size
            hist = get_histogram(ft_img, d, cell_row, cell_col, cell_size)
            hist_list.append(hist)

    assert mode in ["mean", "concat"]
    if mode == "mean":
        features = np.mean(np.stack(hist_list, axis=0), axis=0)
    elif mode == "concat":
        features = np.concatenate(hist_list, axis=0)
    return features

class Features():

    def __init__(self, extractor, use_hist=True, hist_mode="mean", cell_size=16):
        self.extractor = extractor
        self.use_hist = use_hist
        self.cell_size = cell_size
        self.hist_mode = hist_mode

    def __call__(self, img):

        ft_img = self.extractor(img)
        
        if self.use_hist:
            features = process_hists(ft_img, self.extractor.d, 
                            mode=self.hist_mode, cell_size=self.cell_size)
        else:
            features = ft_img.flatten()

        return features




# ----- DATA HANDLING AND EVALUATION -----

def load_data_and_extract(data_dir, feature, shuffle=True, img_size=128):

    paths = [os.path.join(data_dir, fn) for fn in list(os.listdir(data_dir))]
    folder_paths = [pth for pth in paths if os.path.isdir(pth)]
    num_cls = len(folder_paths)

    ft_list = []
    target_list = []

    for i in range(0, num_cls):
        folder_path = folder_paths[i]
        img_fns = [fn for fn in list(os.listdir(folder_path)) if fn.rsplit(".")[-1] == "png"]

        for img_fn in img_fns:
            img_path = os.path.join(folder_path, img_fn)

            # load image
            img = load_image(img_path, h=img_size, w=img_size)
            # extract features
            ft = feature(img)
            
            ft_list.append(ft)
            target_list.append(i+1)

        print(f"{len(target_list)} done...")

    X = np.stack(ft_list, axis=0)
    y = np.array(target_list)

    if shuffle:
        perm = np.random.RandomState(42).permutation(len(y))
        X = X[perm, :]
        y = y[perm]

    return X, y


def cosine_sim_mat(batch, eps=1e-10):
    # given a batch of vectors (n*d), return n*n matrix of pairwise similaries
    n, d = batch.shape
    dot_prod = np.matmul( batch, batch.transpose() )
    lengths = np.linalg.norm(batch, axis=1)
    lengths_mat = np.repeat( np.expand_dims(lengths, 1), n, axis=1)
    lengths_mul = lengths_mat * lengths_mat.transpose()

    cosine_sim = dot_prod / lengths_mul
    return cosine_sim

def cosine_sim_mat_naive(batch):

    n, d = batch.shape

    sim_mat = np.ndarray((n, n))

    for i in range(n):
        for j in range(n):
            a, b = batch[i, :], batch[j, :]
            len_a = np.linalg.norm(a)
            len_b = np.linalg.norm(b)
            dot_prod = np.dot(a, b)
            sim_mat[i, j] = dot_prod / (len_a*len_b)

        if i%100 == 0: print(f"{i} done...")

    return sim_mat

def euclidean_sim_mat_naive(batch):

    n, d = batch.shape

    sim_mat = np.ndarray((n, n))

    for i in range(n):
        for j in range(n):
            a, b = batch[i, :], batch[j, :]
            eucl = np.linalg.norm(a - b)
            sim_mat[i, j] = -eucl

        if i%100 == 0: print(f"{i} done...")

    return sim_mat


def recognition_rate(sim_mat, targets):

    nearest_nbh = np.argsort(sim_mat, axis=1)[:, 1]

    nearest_targets = targets[nearest_nbh]
    num_correct = (targets == nearest_targets).astype(int).sum()
    rate = num_correct / len(targets)

    return rate, nearest_targets

def compute_results(data_dir, features, sim_metrics, shuffle=True, img_size=128):

    results = {}

    for ftname in features:

        print(f"Extracting features with {ftname}...")
        X, y = load_data_and_extract(data_dir, features[ftname])

        rates = {}
        for simname in sim_metrics:

            print(f"Computing rank-1 recognition rate with {simname} similarity...")
            sim_mat = sim_metrics[simname](X)
            rate, nn = recognition_rate(sim_mat, y)
            rates[simname] = rate

        results[ftname] = rates

    return results    


if __name__ == "__main__":

    img = load_image("./AWEDataset/awe/002/02.png")

    ex = BinaryPattern(P=8, R=1)
    ft_img = ex(img)
    print(ft_img)

    ex = FastBinaryPattern(P=8, R=1)
    ft_img = ex(img)
    print(ft_img)

    # ft = Features(ex, use_hist=True, cell_size=32)
    # features = ft(img)
    # print(features)


    

    extractors = {
        "fast bpr (P=8, R=1)": Features(
                                        BinaryPattern(P=8, R=1, rot_inv=False),
                                        cell_size=32),
        "scikit bpr (P=8, R=1)": Features( 
                                        BinaryPattern(P=8, R=1, rot_inv=False),
                                        cell_size=32),

    }

    sim_metrics = {
        "cosine": cosine_sim_mat_naive,
        "euclidean": euclidean_sim_mat_naive
    }

    results = compute_results("./AWEDataset/awe", extractors, sim_metrics)
    print(results)

    # X, y = load_data_and_extract("./AWEDataset/awe", ft)
    # sim_mat = euclidean_sim_mat_naive(X)
    # rate, nn = recognition_rate(sim_mat, y)
    # print(rate)

    # example = np.random.randint(0, 10, (5,5))
    # print(example)

    # ft_img = BinaryPattern(rot_inv=True)(example)
    # print(ft_img)

    # ft_img = FastBinaryPattern()(example)
    # print(ft_img)

    # ft_img = ScikitBinaryPattern()(example)
    # print(ft_img)


