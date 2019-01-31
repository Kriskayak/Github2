python"""
Created on Thu Jan 10 22:01:28 2019

@author: Kris
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
#import robust as rb

num_pixels = 2048 #per axis
star_photons = 3000. #entering aperature/second
pixel_size = .608 #arseconds/pixel
efficiency = 0.92 * 0.7 #camera efficiency * quantum efficiency
time = 2 #initial animation time in seconds
time_step = 2 #time step interval for animation
int_time = 30 #total time in seconds
gain = 3 #photons/electron
bg_photons = 40. #photons/arcsecond
FWHM = 4/.608 #converted from arcseconds to pixels

np.random.seed(150000) #have constant random number set

x, y = np.mgrid[0:num_pixels, 0:num_pixels] #make grid
x0 = y0 = num_pixels // 2 #set center as star location
sigma = FWHM/(2 * np.sqrt(2*np.log(2))) #standard deviation

min_value = int(input("Input lower threshold to plot (default is 0): ") or "0")
max_value = int(input("Input upper threshold to plot (default is 2048): ") or "2048")

fig, ax = plt.subplots(figsize=(10, 8))

def animate(time):

    bg_electrons = np.round(bg_photons * time * efficiency/gain * pixel_size) #total background electrons per pixel^2
    bg_electrons = np.random.poisson(bg_electrons, (num_pixels, num_pixels)) #convert to Poisson array

    star_electrons = np.round(star_photons * time * efficiency/gain) #for the whole star
    Z_amp = star_electrons/(2 * np.pi * sigma**2) #derived from second integral of 2d gaussian fxn
    star = Z_amp * np.exp(-((x-x0)**2 + (y-y0)**2)/(2 * sigma**2)) #create 2d gaussian fxn array

    image = bg_electrons + star

    plt.clf()
    plt.xlim(min_value, max_value) #zoomed view of star
    plt.ylim(min_value, max_value) #zoomed view of star
    plt.text(1940, 1960, time, fontsize=10, color='r')
    plt.imshow(image, vmin=0, vmax=600, cmap='gray', origin='lower')
    plt.colorbar()

#animation
anim = animation.FuncAnimation(fig, animate, frames=np.arange(time, int_time+1, time_step), interval=300, repeat=False)
plt.show()

#new comment on 013019