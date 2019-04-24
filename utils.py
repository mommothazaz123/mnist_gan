import glob
import os
from multiprocessing import JoinableQueue, Process, Queue

import numpy as np
from PIL import Image


def ensure_exists(dir_path):
    try:
        os.makedirs(dir_path)
    except FileExistsError:
        pass
    else:
        print(f"Created directory {dir_path}!")


def load_img(i, o):
    while not i.empty():
        ipath = i.get()
        img = Image.open(ipath)
        x = np.array(img)
        x = (x.astype(np.float32) - 127.5) / 127.5
        o.put(x)
        img.close()
        i.task_done()


def celeba(samples=30000):
    # CelebA: JPEG, 218*178
    # preprocessed to 128*128
    # loads using 3 concurrent processes because of insanity
    path = "datasets/celeba"
    x = []

    q = JoinableQueue()
    r = Queue()

    for ipath in glob.glob(f"{path}/img/*.png")[:samples]:
        q.put(ipath)

    pool = [Process(target=load_img, args=(q, r)) for i in range(3)]
    for p in pool:
        p.start()
    q.join()

    for i in range(r.qsize()):
        x.append(r.get())

    print(f"Loaded data: {len(x)}")
    return np.array(x)


def celeba_64(samples=50000):
    # CelebA: JPEG, 218*178
    # preprocessed to 64*64
    # loads using 3 concurrent processes because of insanity
    path = "datasets/celeba"
    x = []

    q = JoinableQueue()
    r = Queue()

    for ipath in glob.glob(f"{path}/img_64/*.png")[:samples]:
        q.put(ipath)

    pool = [Process(target=load_img, args=(q, r)) for i in range(3)]
    for p in pool:
        p.start()
    q.join()

    for i in range(r.qsize()):
        x.append(r.get())

    print(f"Loaded data: {len(x)}")
    return np.array(x)
