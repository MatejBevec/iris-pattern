import os
import sys
import math
import time
from datetime import datetime

import numpy as np
import scipy as sp
import cv2 as cv
import matplotlib.pyplot as plt


from func import seg_iris_daugman, seg_iris_mediapipe, seg_iris_thuy
from func import draw_iris_seg, get_iris_mask, unwrap_iris, load_img
from func import lbp_extractor, clip_extractor
from func import eucl_dist, kl_divergence
from func import match_pattern, show_matches


# UI RELATED FUNCTIONS

WSIZE = 1000
ISIZE = 400


def capture_img(inp, crop_square=True):
    if isinstance(inp, np.ndarray):
        ret, frame = True, inp
    else:
        ret, frame = inp.read()
    if not ret:
        return None
    img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    if crop_square:
        a = np.min(img.shape[:2])
        sty, stx = ((img.shape[:2]-a) / 2).astype(int)
        img = img[sty:sty+a, stx:stx+a, :]
    return img

def load_imgs(dr, fns, crop_square=True, resize=None):
    if fns is None: fns = sorted(list(os.listdir(dr)))
    imgs = []
    for fn in fns:
        img = load_img(os.path.join(dr, fn), crop_square=crop_square)
        if resize is not None:
            img = cv.resize(img, (resize, resize))
        imgs.append(img)
    return np.array(imgs)

def display_img(img):
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    img = cv.resize(img, (700, 700))
    cv.imshow("iris-patterns", img)
    cv.waitKey(1)

def display_img_plt(img, pl):
    pl.set_data(img)

def iris_limits(iris):
    pupcenter, puprad, ircenter, irrad = iris["pupc"], iris["puprad"], iris["irc"], iris["irrad"]
    x0 = ircenter[0] - irrad
    x1 = ircenter[0] + irrad
    y0 = ircenter[1] - irrad
    y1 = ircenter[1] + irrad
    return y0, y1, x0, x1

def draw_masked_iris(img, iris):
    pupcenter, puprad, ircenter, irrad = iris["pupc"], iris["puprad"], iris["irc"], iris["irrad"]
    mask = get_iris_mask(img, pupcenter, puprad, ircenter, irrad)
    masked = cv.cvtColor(img, cv.COLOR_BGR2BGRA)
    masked[:, :, 3] = mask[:, :, 0]
    y0, y1, x0, x1 = iris_limits(iris)
    print(y0, y1, x0, x1)
    masked = masked[y0:y1, x0:x1, :]
    masked[masked[:, :, 3] == 0] = 255
    masked = masked[:, :, :3]
    return masked

def draw_cropped_iris(img, iris, padding=0):
    y0, y1, x0, x1 = iris_limits(iris)
    y0 = max(y0-padding, 0)
    x0 = max(x0-padding, 0)
    y1 = min(y1+padding, img.shape[0])
    x1 = min(x1+padding, img.shape[0])
    print(x0, x1, y0, y1)
    img = img[y0:y1, x0:x1, :]
    return img

def draw_img_grid(imgs, cols, w, padding, n=None):

    if n is not None:
        imgs = imgs[:n]

    rows = len(imgs) // cols + 1
    imw = int( (w - (cols+1)*padding) / cols )
    h = int( (rows+1)*padding + rows*imw )

    canvas = draw_empty(h, w, 255)

    for i,img in enumerate(imgs):
        img = img[:, :min(img.shape[0], img.shape[1])]
        img = cv.resize(img, (imw, imw))
        row = i // cols
        col = i % cols
        y = (imw+padding) * row + padding
        x = (imw+padding) * col + padding
        canvas[y:y+imw, x:x+imw, :] = img
        
    return canvas

def draw_loadingbar(img, y, x, h, w, ratio, thick):
    img = img.copy()
    y0 = int(y-h/2)
    print(y0)
    x0 = int(x-w/2)
    cv.rectangle(img, (x0, y0), (x0+w, y0+h), (0, 255, 0), thick)
    cv.rectangle(img, (x0, y0), (x0+int(w*ratio), y0+h), (0, 255, 0), -1)
    return img

def draw_loadingiris(img, iris, ratio):
    img = img.copy()
    
    x0, y0 = iris["irc"]
    r = iris["irrad"]

    #angle = angle = 2*math.pi * ratio
    #y1 = y0 + int(r*math.sin(angle))
    #x1 = x0 + int(r*math.cos(angle))
    cv.circle(img, (x0, y0), int(r*ratio), (0,255,0,0.5), -1)
    return img
    
def draw_empty(h, w, val):
    canvas = np.ones((h, w, 3)) * val
    canvas = np.uint8(canvas)
    return canvas

def draw_proc(masked, unwrapped, encoded, matches=None, ratio=1.0):
    """Visualize segmentation, encoding and matching"""

    #ratio = wrapcount/100
    canvas = draw_empty(1000, 1000, 255)

    unwrapped = cv.resize(unwrapped, (750, 250))
    xpos = int(ratio*750)
    unwrapped[:, xpos:, :] = 255
    if True:
        unwrapped[:, xpos-3:xpos, 0] = 255
    canvas[0:250, 250:1000, :] = unwrapped

    masked = cv.resize(masked, (250, 250))
    if True:
        angle = angle = 2*math.pi * ratio
        y = 125 + int(125*math.sin(angle))
        x = 125 + int(125*math.cos(angle))
        masked = cv.line(masked, (125, 125), (x, y), (255, 0, 0), thickness=3)

    canvas[0:250, 0:250, :] = masked

    return canvas

def draw_results(masked, unwrapped, matches, n=3):
    """Visualize best matches alongside query iris"""
    pass


def segment_iris(img):
    pass

def display_loading(img, iris, count):
    pass


# PROCESS OBJECT

class IrisProcess():
    
    def __init__(self, video_path=None, img_path=None, flip=False, pattern_dir="pattern", out_dir="results"):
        self.state = 0
        self.states = [
            self.state_idle,
            self.state_loading,
            self.state_processing,
            self.state_matching,
            self.state_results
        ]
        
        # Used by all states
        self.flip = flip
        self.pattern_dir = pattern_dir
        self.out_dir = out_dir

        self.img = None
        self.cap = None
        if img_path is not None:
            self.cap = cv.imread(img_path) #HACK
        elif video_path is None:
            video_path = cv.VideoCapture(0)
        else:
            self.cap = cv.VideoCapture(video_path)

        self.meshpoints = []
        self.iris = {"irrad": 10} # TEMP
        self.haveiris = False
        self.time = 0 # since calling run() in seconds
        #plt.ion() # TEMP
        
        # Loading
        self.minrad = 10
        self.loaddur = 24 * 3
        self.loadcount = self.loaddur
        
        # Processing
        self.prdur = 24 * 6
        self.prcount = self.prdur

        # Matching
        self.mdur = 24 * 5
        self.mcount = self.mdur

    def every_state(self):
        """Runs regardless of state."""
    
    def state_idle(self):
        """Default state. Capturing footage, segmenting iris."""
        
        # DISPLAY THE CAPTURE

        self.img = capture_img(self.cap)
        if self.flip:
            self.img = cv.rotate(self.img, cv.ROTATE_180)
        iris = seg_iris_mediapipe(self.img) # CHANGE SEGMENTATION ALGO HERE
        if iris:
            self.haveiris = True
            self.iris = {
                "pupc": iris[0],
                "puprad": iris[1],
                "irc": iris[2],
                "irrad": iris[3]
            }

        disp = self.img
        if self.haveiris:
            disp = draw_iris_seg(self.img, self.iris["pupc"], self.iris["puprad"], self.iris["irc"], self.iris["irrad"], (255,0,0))
        display_img(disp)
        
        if self.iris["irrad"] > self.minrad:
            self.state = 1
            return
        
    def state_loading(self):
        """When iris is close enough. Count down until processing."""
        
        # DISPLAY CAPTURE + LOADING

        self.img = capture_img(self.cap)
        if self.flip:
            self.img = cv.rotate(self.img, cv.ROTATE_180)
        iris = seg_iris_mediapipe(self.img) # CHANGE SEGMENTATION ALGO HERE
        if iris:
            self.haveiris = True
            self.iris = {
                "pupc": iris[0],
                "puprad": iris[1],
                "irc": iris[2],
                "irrad": iris[3]
            }

        disp = draw_iris_seg(self.img, self.iris["pupc"], self.iris["puprad"], self.iris["irc"], self.iris["irrad"], (0,255,0))
        ratio = 1 - self.loadcount/self.loaddur
        #disp = draw_loadingbar(disp, self.iris["irc"][1] + 50, self.iris["irc"][0], 20, 100, ratio, 2)
        disp = draw_loadingiris(disp, self.iris, ratio)
        display_img(disp)
        
        if self.loadcount <= 0:
            self.state = 2
            self.loadcount = self.loaddur
            self.start_processing()
            return
        if self.iris["irrad"] < self.minrad:
            self.state = 0
            self.loadcount = self.loaddur
            return
        print(self.loadcount)
        self.loadcount -= 1

    def start_processing(self):
        """Called on transition to processing state"""
        self.fimg = self.img
        self.unwrapped = unwrap_iris(self.img, self.iris["irc"], self.iris["puprad"], self.iris["irrad"])
        h, w, _ = self.unwrapped.shape
        if h < 64:
            self.unwrapped = cv.resize(self.unwrapped, (64, int((w/h)*64)))
        #self.masked = draw_masked_iris(self.img, self.iris)
        self.masked = draw_cropped_iris(self.img, self.iris)
        
    def state_processing(self):
        """After loading completes. Visualize unwrapping."""
        
        # UNROLL IRIS

        ratio = 1 - self.prcount/self.prdur
        disp = draw_proc(self.masked, self.unwrapped, None, ratio=ratio)
        display_img(disp)
        
        if self.prcount <= 0:
            self.state = 3
            self.prcount = self.prdur
            self.start_matching()
        self.prcount -= 1

    def start_matching(self):
        """"Called on transition to matching state."""

        print("Loading patterns and extracting features...")
        self.fns, self.dists, self.besti = match_pattern(self.unwrapped, self.pattern_dir,
                                            extractor=lbp_extractor, dist_func=eucl_dist, resize=800)
        self.imgs = []
        self.mi = 0
        self.maxdist = np.max(self.dists)
        #show_matches(self.unwrapped, self.pattern_dir, self.fns, self.dists, self.besti)

    def state_matching(self):
        """After processing. Visualize encoding and distance comp"""

        # LOAD PATTERNS
        # ENCODE IRIS AND PATTERNS

        ratio = 1 - self.mcount/self.mdur
        disp = draw_proc(self.masked, self.unwrapped, None, ratio=1)

        if self.mcount % 1 == 0 and self.mi < len(self.fns):
            print(self.mcount, self.mi)
            img = load_img(os.path.join(self.pattern_dir, self.fns[self.mi]))
            dist_ratio = math.sqrt(self.dists[self.mi] / self.maxdist)
            print(dist_ratio)
            bg = draw_empty(img.shape[0], img.shape[1], 255)
            img = cv.addWeighted(bg, dist_ratio, img, 1-dist_ratio, 0)
            img = cv.resize(img, (ISIZE, ISIZE))
            self.imgs.append(img)
            self.mi += 1 

        grid = draw_img_grid(np.array(self.imgs), 10, 1000, 10)
        disp[250:grid.shape[0]+250, :grid.shape[1], :] = grid
        display_img(disp)

        if self.mi >= len(self.fns):
            self.state = 4
            self.mcount = self.mdur
            self.start_results()
        self.mcount -= 1
        pass

    def start_results(self):
        """Called on transition to results state."""

        #disp = draw_proc(self.masked, self.unwrapped, None, ratio=1)
        disp = draw_empty(1000, 1000, 255)
        eye = draw_cropped_iris(self.img, self.iris, 25)
        eye = cv.resize(eye, (500, 500))
        disp[100:600, 250:750, :] = eye
        imgs = load_imgs(self.pattern_dir, self.fns, resize=500)
        ordered = imgs[self.besti]
        grid = draw_img_grid(ordered, 3, 1000, 50, n=3)
        disp[600:, 0:, :] = grid[:400, :, :]
        display_img(disp)

        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M")
        path = os.path.join(self.out_dir, dt_string + ".png")
        print(path)
        cv.imwrite(path, cv.cvtColor(disp, cv.COLOR_RGB2BGR))

        # TEMP?
        #input("Press any key to restart...")
        cv.waitKey(0)
        self.state = 0
        
    def state_results(self):
        """After processing completes. Present matches."""
        pass
    
    def run(self):

        self.time = 0
        while(True):
            self.time = time.time()
            
            self.every_state()
            self.states[self.state]()

            if cv.getWindowProperty("iris-patterns", 0) < 0:
                exit()

            elapsed = time.time() - self.time
            tosleep = max(1/24 - elapsed, 0)
            #time.sleep(tosleep) # BUG here?



if __name__ == "__main__":    
    pr = IrisProcess(video_path=0, flip=False)
    pr.run()

    # fns = [f"{i}.jpg" for i in range(1,68) if i != 4]
    # print(fns)
    # imgs = load_imgs("pattern", fns, resize=200)
    # grid = draw_img_grid(imgs, 12, 1000, 10)
    # grid = cv.cvtColor(grid, cv.COLOR_RGB2BGR)
    # cv.imwrite("dataset.png", grid)