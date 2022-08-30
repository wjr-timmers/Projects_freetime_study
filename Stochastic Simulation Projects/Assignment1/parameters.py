# Image size (pixels)
WIDTH = 1024
HEIGHT = 1024

# Plot window
RE_START = -2.0 # x as
RE_END = 0.47
IM_START = -1.12  # y as
IM_END = 1.12
AREA_SQUARE = (abs(RE_START) + abs(RE_END)) * (abs(IM_START) + abs(IM_END))

NO_SAMPLES = 10000   # for orthogonal sampling make sure the root of this value is an integer as well
MAX_ITER = 1000
STAT_SAMPLE_SIZE = 1000

# confidence interval
alpha = 0.95

# Beta distribution target parameters, beta fixed at 1.0
a = 2
a2 = 2.4