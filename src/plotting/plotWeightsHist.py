import sys
import numpy as np
import matplotlib.pyplot as plt

f = sys.argv[1]+'/final_convnet_state.pkl'
data = np.load(f)
data = data['best_params']
for i in range(len(data)/2):
    mat = data[i*2]
    mat = mat.flatten()
    plt.hist(mat, bins=30)
    plt.title('index '+str(i*2))
    plt.show()
