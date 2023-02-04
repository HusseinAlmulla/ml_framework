import math


# this is base one the experiment of the paper https://arxiv.org/pdf/1708.04321.pdf
#  the code is in https://github.com/HusseinAlmulla/hassanat-distance-checker
#  there are more distance function in the paper

# Hassanat
def HasD(x, y):
    total = 0
    for xi, yi in zip(x, y):
        min_value = min(xi, yi)
        max_value = max(xi, yi)
        total += 1  # we sum the 1 in both cases
        if min_value >= 0:
            total -= (1 + min_value) / (1 + max_value)
        else:
            # min_value + abs(min_value) = 0, so we ignore that
            total -= 1 / (1 + max_value + abs(min_value))
    return total


# Lorentzian distance
def LD(x, y):
    total = 0
    for xi, yi in zip(x, y):
        total += math.log(1 + abs(xi - yi))
    return total
