import sys
import numpy as np
import matplotlib.pyplot as plt

f = sys.argv[1]+'/parameters.npy'
data = np.load(f)
for i in range(len(data)/2):
    mat = data[i*2]
    mat = mat.get_value().flatten()
    plt.hist(mat, bins=30)
    plt.title('index '+str(i*2))
    plt.show()
