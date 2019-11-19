import os
import time
from skimage import io, img_as_float
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

start = time.time()

gaussian_sigma = 64
kernel_normal = 64

project_path = '/Volumes/SSD/PROJECTS'
project_name = 'SUB_2'


def pre_proc(img, gaussian_sigma, kernel_normal):
    from skimage import util
    from skimage import filters
    from skimage import exposure
    from skimage import img_as_float
    import numpy as np
    import warnings

    # convert to 16-bit
    warnings.filterwarnings("ignore", category=UserWarning)
    img_f = img_as_float(img)

    # equalize histogram
    img_max = np.max(img_f.flatten())
    img_min = np.min(img_f.flatten())
    img2 = (img - img_min) / (img_max - img_min)

    # invert image
    img3 = util.invert(img2)

    # subtract gaussian blur
    background = filters.gaussian(img3, sigma=gaussian_sigma, preserve_range=True)
    img4 = img3 - background

    # Adaptive Equalization
    img5 = exposure.equalize_adapthist(img4, kernel_size=kernel_normal, clip_limit=0.01, nbins=256)

    return img5


f = 1
r = [0, 1]

f1 = img_as_float(io.imread(os.path.join(project_path, project_name, 'EXP', 'CORRECTED', 'IMG_'+str(f)+'.tif')))
f2 = img_as_float(io.imread(os.path.join(project_path, project_name, 'EXP', 'CORRECTED', 'IMG_' + str(f+1) + '.tif')))
fr = [f1, f2]

frp = Parallel(n_jobs=2)(delayed(pre_proc)(i, gaussian_sigma, kernel_normal) for i in fr)

f1p = frp[0]
f2p = frp[1]

end = time.time()
print(' ')
print('> Processing done in '+str(end - start)+' seconds')

fig = plt.figure()
plt.imshow(f1p)
plt.show()
