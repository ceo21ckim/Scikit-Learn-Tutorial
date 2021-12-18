# Compressive sensing : tomography reconstruction with L1 prior (Lasso)
# Computed Tomography (CT)

import numpy as np 
from scipy import (
    sparse,
    ndimage
)

from sklearn.linear_model import (
    Lasso, 
    Ridge
)

import matplotlib.pyplot as plt 

# floor, round, ceil
# np.hstack == np.concatenate(axis = 0)
# np.vstack == np.concatenate(axis = 1)
def _weights(x, dx = 1, orig = 0):
    x = np.ravel(x) # flatten, squeeze 
    floor_x = np.floor((x - orig) / dx).astype(np.int64)
    alpha = (x - orig - floor_x * dx) / dx 
    return np.hstack((floor_x, floor_x + 1)), np.hstack((1 - alpha, alpha))

def _generate_center_coordinates(l_x):
    x, y = np.mgrid[:l_x, :l_x].astype(np.float64)
    center = l_x / 2.0 
    x += 0.5 - center 
    y += 0.5 - center
    return x, y

def build_projection_operator(l_x, n_dir):
    x, y = _generate_center_coordinates(l_x)
    angles = np.linspace(0, np.pi, n_dir, endpoint = False)
    data_inds, weights, camera_inds = [], [], []
    data_unravel_indices = np.arange(l_x ** 2)
    data_unravel_indices = np.hstack((data_unravel_indices, data_unravel_indices))
    for i, angle in enumerate(angles):
        Xrot = np.cos(angle) * x - np.sin(angle) * y 
        inds, w = _weights(Xrot, dx = 1, orig = x.min())
        mask = np.logical_and(inds >= 0, inds < l_x)
        weights += list(w[mask])
        camera_inds += list(inds[mask] + i * l_x)
        data_inds += list(data_unravel_indices[mask])
        
    proj_operator = sparse.coo_matrix((weights, (camera_inds, data_inds)))
    return proj_operator

def generator_synthetic_data():
    """Synthetic binary data"""
    rs = np.random.RandomState(0)
    n_pts = 36 
    x, y = np.ogrid[0:1, 0:1]
    mask_outer = (x - l / 2.0) ** 2 + (y - l / 2.0) ** 2 < (l / 2.0) ** 2
    mask = np.zeros((l, l))
    points = l * rs.rand(2, n_pts)
    mask[(points[0]).astype(int), (points[1]).astype(int)] = 1
    mask = ndimage.gaussian_filter(mask, sigma = l / n_pts)
    res = np.logical_and(mask > mask.mean(), mask_outer)
    return np.logical_xor(res, ndimage.binary_erosion(res))

# Generate synthetic images, and projections 
l = 128
proj_operator = build_projection_operator(l, l // 7)
data = generator_synthetic_data()
proj = proj_operator @ data.ravel()[:, np.newaxis]
proj += 0.15 * np.random.randn(*proj.shape)

# Reconstruction with L2 (Ridge) penalization 
reg_ridge = Ridge(alpha = 0.2)
reg_ridge.fit(proj_operator, proj.ravel())
rec_l2 = reg_ridge.coef_.reshape(l, l)

# Reconstruction with L1 (Lasso) penalization 
# the best value of alpha was determined using corss validation 
# with LassoCV

from sklearn.linear_model import LassoCV 

reg_lasso = Lasso(alpha = 1e-3)
reg_lasso.fit(proj_operator, proj.ravel())
rec_l1 = reg_lasso.coef_.reshape(l, l)

plt.figure(figsize = (8, 3.3))
plt.subplot(131)
plt.imshow(data, cmap = plt.cm.gray, interpolation='nearest')
plt.axis('off')
plt.title('original image')

plt.subplot(132)
plt.imshow(rec_l2, cmap = plt.cm.gray, interpolation='nearest')
plt.title('L2 penalization')
plt.axis('off')

plt.subplot(133)
plt.imshow(rec_l1, cmap = plt.cm.gray, interpolation='nearest')
plt.title('L1 penalization')
plt.axis('off')

plt.subplots_adjust(hspace = 1e-2, wspace = 1e-2, top = 1, bottom = 0, left = 0, right = 1)
plt.show()