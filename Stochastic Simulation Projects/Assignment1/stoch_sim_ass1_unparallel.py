from PIL import Image, ImageDraw
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,6))


# code from https://www.codingame.com/playgrounds/2358/how-to-plot-the-mandelbrot-set/mandelbrot-set
# https://www.fractalus.com/kerry/articles/area/mandelbrot-area.html , even checken of we dit wel mogen gebruiken

# Set seed for reproducibility
np.random.seed(420)


def mandelbrot(c, max_iter):
    z = 0
    n = 0
    while abs(z) <= 2 and n < max_iter:
        z = z * z + c
        n += 1
    return n


# Image size (pixels)
WIDTH = 1024
HEIGHT = 1024

# Plot window
RE_START = -2.0
RE_END = 0.47
IM_START = -1.12
IM_END = 1.12
AREA_SQUARE = (abs(RE_START) + abs(RE_END)) * (abs(IM_START) + abs(IM_END))


def monte_carlo(n=0, max_iter=0):
    sample_area_count = 0
    for i in tqdm(range(n)):  # tqdm used to plot progress bar
        # Draw random numbers from uniform distribution
        x = np.random.uniform(low=0, high=1)
        y = np.random.uniform(low=0, high=1)

        # Convert coordinates to complex number and compute the mandelbrot iterations
        c = complex(RE_START + x * (RE_END - RE_START),
                    IM_START + y * (IM_END - IM_START))
        # Compute the number of iterations
        m = mandelbrot(c, max_iter)
        # The color depends on the number of iterations
        color = 255 - int(m * 255 / max_iter)
        if color == 0:
            sample_area_count += 1
            plt.plot(RE_START + x * (RE_END - RE_START), IM_START + y * (IM_END - IM_START), 'ro', color='green', ms=72. / fig.dpi)
        else:
            plt.plot(RE_START + x * (RE_END - RE_START), IM_START + y * (IM_END - IM_START), 'ro', color='red', ms=72. / fig.dpi)

    # Caculate average of the samples
    within_mb_set = sample_area_count / n
    area_approx = within_mb_set * AREA_SQUARE
    plt.title('Monte Carlo approximation')
    plt.savefig('montecarlo.jpg')
    plt.show()
    return area_approx


area = monte_carlo(n=50000, max_iter=200)
print('Area approximated at: ', area)