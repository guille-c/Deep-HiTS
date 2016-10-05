import numpy as np
import matplotlib.pyplot as plt
import sys

filename = sys.argv[1]
f = open(filename,"r")

iters = []
loss = []
for row in f:
    if ", validation loss" in row:
        elems = row.strip().split()
        loss.append(elems[-1])
        iters.append(elems[3][:-1])

plt.plot(iters, loss)
plt.xlabel('Iteration')
plt.ylabel('loss (cross-entropy)')
plt.title('Learning curve (validation set)')
plt.show()
