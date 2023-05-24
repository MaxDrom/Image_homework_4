from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import Gaussian2DKernel
from photutils.segmentation import detect_sources
from photutils.segmentation import detect_threshold
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.convolution import convolve
from photutils.segmentation import deblend_sources
from scipy.optimize import minimize, least_squares
import math

def pmin(f, p, data):
    x = np.array([[i for j in range(data.shape[1])] for i in range(data.shape[0])])
    y = np.array([[j for j in range(data.shape[1])] for i in range(data.shape[0])])
    min_func = lambda pars: np.reshape( f(x, y, pars) - data, (data.shape[1]*data.shape[0], ))
    return least_squares(min_func, p).x


def gaussian(x, y, pos, sigma1, sigma2, I, angle):
    new_x = (x-pos[0])*math.cos(angle) - (y-pos[1])*math.sin(angle)
    new_y = (x-pos[0])*math.sin(angle) + (y-pos[1])*math.cos(angle)
    return I*np.exp(-new_x**2/(2*sigma1*sigma1) -new_y**2/(2*sigma2*sigma2))


def gaussian_int(sigma1, sigma2, I):
    return 2*math.pi*I*sigma1*sigma2

with fits.open("data.fits") as hdu:
    data = hdu[0].data

sigma = 3.0*gaussian_fwhm_to_sigma
kernel = Gaussian2DKernel(sigma, x_size = 3, y_size = 3)
kernel.normalize()
threshold = detect_threshold(data, nsigma = 15.)
convolved_data = convolve(data, kernel)
segm = detect_sources(convolved_data, threshold, connectivity=8, npixels = 1)
segm_deblend = deblend_sources(convolved_data, segm, npixels=1, nlevels=1024, contrast=0, progress_bar = False).data
positions = []
intensity = []
sigma = []
angle = []
n =  int(np.max(segm_deblend))
for i in range(1, n+1):
    star_data = np.where(segm_deblend == i, data, 0)
    position = np.unravel_index(np.argmax(star_data), data.shape)
    positions = [*positions,*position]
    intensity.append(data[position])
    sigma.append(1.)
    sigma.append(1.)
    angle.append(0)

p = [*positions, *sigma, *intensity, *angle ]

def model(x, y,p):
    result = 0
    for i in range(n):
        result = result+gaussian(x, y, [p[2*i], p[2*i+1]], p[2*n+2*i], p[2*n+2*i+1], p[4*n+i],
                                 p[5*n+i])
    return result

p = pmin(model, p, data)
for i in range(n):
    print(f"Flux: {gaussian_int(p[2*n+2*i], p[2*n+2*i+1], p[4*n+i])}")
    print(f"Position: ({p[2*i]}, {p[2*i+1]}), Intensity: {p[4*n+i]}, Sigma_x: {p[2*n+2*i]},  Sigma_y: {p[2*n+2*i+1]}, Angle: {(p[5*n+i]/2/math.pi-math.floor(p[5*n+i]/2/math.pi))*360}")
x = np.array([[i for j in range(data.shape[1])] for i in range(data.shape[0])])
y = np.array([[j for j in range(data.shape[1])] for i in range(data.shape[0])])
new_data = model(x, y, p)
fig, ax = plt.subplots()
ax.imshow(new_data, origin="lower")
fig.set_figwidth(data.shape[0]/10)   
fig.set_figheight(data.shape[1]/10)    
fig.savefig("model_image.png")