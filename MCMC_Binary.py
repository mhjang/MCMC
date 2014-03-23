__author__ = 'mhjang'

import numpy as np
import math
import random

original_img = np.genfromtxt('stripes.txt')
noisy_img = np.genfromtxt('stripes-noise.txt')

def generateSample(t, W_p, W_l):
    # initialize with y_ij = x_ij
    samples = list()
    maes = list()

    for k in range(t):
        y = np.copy(noisy_img)
        for i in range(100):
            for j in range(100):
                neighbors = get_neighbors((i,j))
                # len([y_lk = 0]
                neighbors_zero = len(neighbors) - sum([y[k][l] for (k, l) in get_neighbors((i,j))])
                norm = math.exp(W_p*(len(neighbors) - neighbors_zero) + W_l * noisy_img[i][j])
                denorm = math.exp(W_p * neighbors_zero + W_l * (1-noisy_img[i][j]))
                + norm
                prob_y_ij = norm / denorm
                alpha = random.uniform(0, 1)
                if alpha < prob_y_ij:
                    y[i][j] = 0
                else:
                    y[i][j] = 1
        samples.append(y)
        maes.append(calMAE(sum(samples)/(k+1)))
   #     print(y)
    print(maes)
    return sum(samples)/t


# given a pixel coordinate (i, j), return a list of coordinates of its neighbors
def get_neighbors(coord):
    i = coord[0]
    j = coord[1]
    # if it's on the border of the top
    # (i, j-1), (i-1, j), (i, j+1), (i+1, j)
    neighbors = list()
    if is_in_range(i):
        if is_in_range(j-1):
            neighbors.append((i, j-1))
        if is_in_range(j+1):
            neighbors.append((i, j+1))
    if is_in_range(j):
        if is_in_range(i-1):
            neighbors.append((i-1, j))
        if is_in_range(i+1):
            neighbors.append((i+1, j))
    return neighbors
def is_in_range(i):
    if i>=0 and i<100:
        return True
    else:
        return False


def calMAE(y):
    error = 0
    for i in range(100):
        for j in range(100):
            error += math.fabs(y[i][j] - original_img[i][j])
    print("error" + str(error / (100*100)))
    return error/(100*100)

def main():
    W_p = 30
    W_l = 300
    s = generateSample(100, W_p, W_l)
   # print(str(W_p) + ", " + str(w_l))
 #   calMAE(s)
  #  s = generateSample(100, 0, 178)
  #  calMAE(s)


if __name__ == '__main__':
    main()