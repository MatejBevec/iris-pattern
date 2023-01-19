import os
import sys
import math
import time

import numpy as np
import scipy as sp
import cv2

from IPython.display import display
from IPython.core.pylabtools import figsize
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

from lbp2 import ScikitBinaryPattern, Features
from skimage.feature import local_binary_pattern as lbp_scikit
import mediapipe as mp
facemesh = mp.solutions.face_mesh

from daugman import find_iris
from segment import segment



# ------- UTILITIES --------

def load_img(pth, crop_square=True):
    img = cv2.cvtColor(cv2.imread(pth), cv2.COLOR_BGR2RGB)
    if crop_square:
        a = np.min(img.shape[:2])
        sty, stx = ((img.shape[:2]-a) / 2).astype(int)
        img = img[sty:sty+a, stx:stx+a, :]
    return img

def draw_iris_seg(img, pupcenter, puprad, ircenter, irrad, color=(255, 0, 0)):
    out = img.copy()
    ircenter = pupcenter
    thick = int(img.shape[0]/500)+1
    cv2.circle(out, ircenter, irrad, color, thick)
    cv2.circle(out, pupcenter, puprad, color, thick)
    return out

def show_iris_seg(img, pupcenter, puprad, ircenter, irrad):
    out = draw_iris_seg(img, pupcenter, puprad, ircenter, irrad)
    plt.imshow(out)
    plt.show()


    
def get_iris_mask(img, pupc, puprad, irc, irrad):
    
    mask = np.zeros_like(img)
    irmask = cv2.circle(mask.copy(), irc, irrad, (255, 255, 255), -1)
    pupmask = cv2.circle(mask.copy(), pupc, puprad, (255, 255, 255), -1)
    mask = cv2.subtract(irmask, pupmask)
    
    return mask

def unwrap_iris(img, center, puprad, irrad):
    
    w = irrad - puprad
    steps = int(2 * math.pi * puprad)
    
    out = np.zeros((w, steps, 3))
    
    for i in range(steps):
        for j in range(w):
            angle = 2*math.pi * (i/steps)
            r = puprad + j - 1
            y = center[1] + int(r*math.sin(angle))
            x = center[0] + int(r*math.cos(angle))
            out[j, i, :] = img[y, x, :]
    
    return out.astype(np.uint8)

    
    
# ------- SEGMENTATION ALGORITHMS (find the iris region) --------

def seg_iris_daugman(square_img, resize=200):
    """Daugman iris segmentation algoritm"""
    
    # SLOW AND UNRELIABLE atm !!!
    
    imgsmall = square_img
    if resize is not None:
        RATIO = square_img.shape[0] / resize
        imgsmall = cv2.resize(square_img, (resize, resize))

    gray = cv2.cvtColor(imgsmall, cv2.COLOR_RGB2GRAY)
    
    minr, maxr = int(0.15*gray.shape[0]), int(0.5*gray.shape[0])
    points_step = int(gray.shape[0] / 100)
    dg_step = max(1, int(points_step / 2))
    
    ircenter, irrad = find_iris(gray, daugman_start=minr, daugman_end=maxr, daugman_step=dg_step, points_step=points_step)
    pupcenter, puprad = find_iris(gray, daugman_start=5, daugman_end=irrad, daugman_step=dg_step, points_step=points_step)
    
    if resize is not None:
        ircenter = (int(ircenter[0]*RATIO), int(ircenter[1]*RATIO))
        pupcenter = (int(pupcenter[0]*RATIO), int(pupcenter[1]*RATIO))
        irrad = int(irrad*RATIO); puprad = int(puprad*RATIO)

    return pupcenter, puprad, ircenter, irrad, square_img



LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [429, 470, 471, 472]

def seg_iris_mediapipe(img):
    """Segmentation with Google's mediapipe stack"""
    
    # FAST AND RELIABLE
    # BUT ONLY WORKS IF ENTIRE FACE IS SEEN

    w, h = img.shape[:2]
    
    with facemesh.FaceMesh(max_num_faces=1,
                            refine_landmarks=True,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5) as fm:
        results = fm.process(img)
        if results.multi_face_landmarks:
            #print("face mesh detected")
            lms = results.multi_face_landmarks
            meshpoints = np.array([np.multiply([p.x, p.y] , [w, h]).astype(int) for p in lms[0].landmark])
            
            left_iris = meshpoints[LEFT_IRIS]
            right_iris = meshpoints[RIGHT_IRIS]
            
            
            #img = cv2.polylines(img, [left_iris], True, (255, 0, 0), 1, cv2.LINE_AA)
            
            center = np.mean(left_iris, axis=0)
            rad = np.mean(np.linalg.norm(left_iris - center, axis=1))
            center = center.astype(int)
            irrad = int(rad)
            
            puprad = int(0.3 * irrad)
            
            #img = cv2.circle(img, center, irrad, (0, 255, 0), 3)

        else:
            print("No facemesh detected (probably too close to camera)...")
            return None
        
        #figsize(6, 6)
        #plt.imshow(img)
        
        return center, puprad, center, irrad, img
    


def seg_iris_thuy(img, resize=None):
    """Another Daugman detector implementation"""
    
    imgsmall = img
    if resize is not None:
        RATIO = img.shape[0] / resize
        imgsmall = cv2.resize(img, (resize, resize))
    
    img_gray = cv2.cvtColor(imgsmall, cv2.COLOR_RGB2GRAY)
    
    iris, pupil, img_masked = segment(img_gray, eyelashes_thres=10, use_multiprocess=False)
    irx, iry, irrad = iris
    pupx, pupy, puprad = pupil
    pupcenter, ircenter = (pupy, pupx), (iry, irx)
    
    if resize is not None:
        ircenter = (int(ircenter[0]*RATIO), int(ircenter[1]*RATIO))
        pupcenter = (int(pupcenter[0]*RATIO), int(pupcenter[1]*RATIO))
        irrad = int(irrad*RATIO); puprad = int(puprad*RATIO)
        img_masked = cv2.resize(img_masked, (img.shape[1], img.shape[0]))
    
    return ircenter, puprad, pupcenter, irrad, img_masked





# ------- FEATURE EXTRACTION AND MATCHING METHODS ---------


from sentence_transformers import SentenceTransformer
from PIL import Image

def _get_histogram(ft_img, d, row, col, cell_size):
    """Compute LBP histogram for one cell"""
    
    cell = ft_img[row:row+cell_size, col:col+cell_size]
    hist = np.zeros(d).astype(int)

    unique, counts = np.unique(cell, return_counts=True)
    #print(unique)
    #print(counts)
    unique = unique.astype(int)
    hist[unique] = counts
    
    #print(cell[:10, :10])
    return hist

def _process_hists(ft_img, d, cell_size=16, mode="mean"):
    """Compute histogramed feature vector from a LBP image"""

    hist_list = []
    rows = int( ft_img.shape[0] / cell_size )
    cols = int( ft_img.shape[1] / cell_size )
    
    for i in range(rows):
        cell_row = i*cell_size

        for j in range(cols):
            cell_col = j*cell_size
            hist = _get_histogram(ft_img, d, cell_row, cell_col, cell_size)
            hist_list.append(hist)

    assert mode in ["mean", "concat"]
    if mode == "mean":
        features = np.mean(np.stack(hist_list, axis=0), axis=0)
    elif mode == "concat":
        features = np.concatenate(hist_list, axis=0)
    return features


# Extractors:

def lbp_extractor(img, d=8*3, r=3, cell_size=32):
    """Extract Linear Binary Pattern (LBP) features from image"""
    
    if len(img.shape) > 2:
        img = img[:, :, 0]
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    ftimg = lbp_scikit(img, d, r, "uniform").astype(int)
    ft = _process_hists(ftimg, d+2, cell_size=cell_size, mode="mean")
    return ft


"""Extract features from an image with CLIP neural embedding"""
clip = SentenceTransformer("clip-ViT-B-32")
#clip_extractor = lambda img: clip.encode(Image.fromarray(img))

def clip_extractor(img):
    return clip.encode(Image.fromarray(img))


# Similarity metrics:

def kl_divergence(p, q):
    """Compute KL-divergence between two feature vectors"""
    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))

"""Compute eucledian distance between two feature vectors"""
#eucl_dist = lambda x, y: np.linalg.norm((x-y)**2)

def eucl_dist(x, y):
    return np.linalg.norm((x-y)**2)



#  ------ FIND MATCHES AND PRESENT RESULTS -------


# quick test


def match_pattern(q, pt_dir, resize=400,
                    extractor=lbp_extractor, 
                    dist_func=kl_divergence,
                    verbose=True,
                    cached=True):
    """Compare query image (iris) to images from pt_dir (patterns) in terms of extracted features.
        Find and return best matches.
    """
    
    fns, dists = [], []
    if verbose: print("Extracting query features..")
    q_ft = extractor(q)
    
    #print(len(q_ft), q_ft[:10])

    cache_pth = os.path.join("cache", f"{os.path.split(pt_dir)[-1]}_{extractor.__name__}.npy")
    loaded = None
    if os.path.isfile(cache_pth) and cached:
        loaded = np.load(cache_pth)

    if verbose:
        if loaded is None: print("Extracting database features...")
        else: print("Loading precomputed features...")
    
    features = []

    for i,fn in enumerate(os.listdir(pt_dir)):
        fns.append(fn)
        if loaded is None:
            pattern = load_img(os.path.join(pt_dir, fn), crop_square=False)
            if resize is not None: pattern = cv2.resize(pattern, (resize, resize))
            pattern_ft = extractor(pattern)
        else:
            pattern_ft = loaded[i, :]
        features.append(pattern_ft)
        dist = dist_func(q_ft, pattern_ft)
        dists.append(dist)
        if verbose: print(fn, dist)
    besti = np.argsort(dists)

    if cached:
        np.save(cache_pth, features)
        
    return fns, dists, besti
    
    
def show_matches(q, pt_dir, fns, dists, besti, n=3):
    """Display matching steps for one query image"""
    print(f"Best match = {fns[besti[0]]}")
    figsize(10,3)
    plt.imshow(q)
    plt.axis('off')
    plt.show()
    n = 3
    figsize(10,4)
    for i, bi in enumerate(besti[:n]):
        match = load_img(os.path.join(pt_dir, fns[bi]), crop_square=True)
        plt.subplot(1, n, i+1)
        plt.imshow(match)
        plt.axis('off')
    plt.show()
    
def show_matches_subfig(q, pt_dir, fns, dists, besti, k=3, q2=None):
    """Display matches in compact form"""
    if q2 is not None: k += 1
    figsize(12,3)
    plt.subplot(1, k+1, 1)
    q = q[:, :min(q.shape[0], q.shape[1])]
    plt.imshow(q)
    plt.axis('off')
    if q2 is not None:
            plt.subplot(1, k+1, 2)
            q2 = q2[:, :min(q.shape[0], q.shape[1])]
            plt.imshow(q2)
            plt.axis('off')
    for i, bi in enumerate(besti[:k-1]):
        match = load_img(os.path.join(pt_dir, fns[bi]), crop_square=True)
        plt.subplot(1, k+1, i+3)
        plt.imshow(match)
        plt.axis('off')
        
def show_matches_subfig2(q,pt_dir, fns, dists, besti):
    figsize(6,9)
    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig)
    ax = fig.add_subplot(gs[:, 0])
    q = q[:, :min(q.shape[0]*2, q.shape[1])]
    q = np.transpose(q, (1,0,2))
    ax.imshow(q)
    ax.axis('off')
    for i, bi in enumerate(besti[:4]):
        match = load_img(os.path.join(pt_dir, fns[bi]), crop_square=True)
        print(i+2+ (i//3))
        ax = fig.add_subplot(gs[(i//2)+1:, i%2])
        #plt.subplot(2, 3, i+2+ )
        ax.imshow(match)
        ax.axis('off')
    plt.tight_layout()



if __name__ == "__main__":


    img = load_img("iris/3.jpg")
    plt.imshow(img); plt.show()

    pupcenter, puprad, ircenter, irrad, _  = seg_iris_daugman(img)
    show_iris_seg(img, pupcenter, puprad, ircenter, irrad)

    unwrapped = unwrap_iris(img, pupcenter, puprad, irrad)
    plt.show(); plt.imshow(unwrapped); plt.show()

    fns, dists, besti = match_pattern(unwrapped, "pattern", extractor=clip_extractor, dist_func=eucl_dist, resize=400)
    show_matches(unwrapped, "pattern", fns, dists, besti)