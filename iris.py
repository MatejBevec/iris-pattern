import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

from func import load_img, unwrap_iris, match_pattern, show_matches_subfig
from func import seg_iris_daugman, seg_iris_mediapipe
from func import lbp_extractor, clip_extractor, eucl_dist, kl_divergence
from process import IrisProcess

PATTERN_DIR = "pattern"
DIR_OUT_PATH = "matches.png"
APP_OUT_PATH = ""

RESIZE=800


def run_from_dir(dir):
    """Find matches for iris photos in given dir and save a results image."""

    files = list(os.listdir(dir))
    image_rows = []

    print(f"Searching patterns for iris images in {dir}:")

    for f in files:
        #pth = f"{dir}/{f}.jpg"
        pth = os.path.join(dir, f)
        print(f"Finding matches for {f}...")
        img = load_img(pth)
        pupcenter, puprad, ircenter, irrad, _  = seg_iris_daugman(img)
        unwrapped = unwrap_iris(img, pupcenter, puprad, irrad)
        #show_iris_seg(img, pupcenter, puprad, ircenter, irrad)
        fns, dists, besti = match_pattern(unwrapped, PATTERN_DIR,
                                        extractor=lbp_extractor, dist_func=eucl_dist,
                                        verbose=False, resize=RESIZE)
        q2 = unwrapped; q2 = q2[:, :min(q2.shape[0], q2.shape[1])]
        show_matches_subfig(img, PATTERN_DIR, fns, dists, besti, q2=q2)
        #plt.show()

        plt.savefig("temp.png", bbox_inches='tight', pad_inches=0.5)
        row = cv.imread("temp.png")
        image_rows.append(row)

    output = np.concatenate(image_rows, axis=0)
    cv.imwrite(DIR_OUT_PATH, output)

def run_from_cam(video_path=0, img_path=None, flip=False):
    pr = IrisProcess(video_path=video_path, img_path=img_path, flip=flip)
    pr.run()

if __name__ == "__main__":

    if len(sys.argv) < 2 or sys.argv[1] == "cam":
        run_from_cam()
    elif sys.argv[1] == "dir":
        run_from_dir(sys.argv[2])
    elif sys.argv[1] == "video":
        run_from_cam(video_path="iris2.mp4", flip=True)
    elif sys.argv[1] == "img":
        run_from_cam(img_path="face.jpg")