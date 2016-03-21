import sys
import theano
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

f = np.load(sys.argv[1])
#filters = f[-2].get_value()
#print filters.shape
filters = f['best_params'][0]
shape = filters.shape

img_names = ['temp_filters.png','sci_filters.png','diff_filters.png','snr_diff_filters.png']
for chan in range(shape[1]):
        plt.clf()
        low = np.absolute(filters[:,chan,:,:]).min()
        high = np.absolute(filters[:,chan,:,:]).max()
        for filter in range(shape[0]):
                plt.subplot(4,8,filter+1)
                im = np.absolute(filters[filter, chan, :, :])
                im -= low
		im *= 255.0/(high-low+0.001)
                im = im.astype('uint8')
                #print im.dtype
                #print low, high
                #print im.min(), im.max()
                #exit()
		plt.imshow(im, interpolation='none', cmap = cm.Greys_r, vmin=0, vmax=255)#, cmap = cm.Greys_r)
		plt.axis('off')
	plt.savefig(img_names[chan])
exit()

layer_names = ['conv1']#, 'conv2']
layer_nfilters = [16, 50]
for i in range(len(layer_names)):
        plt.clf()
	filters = net.params[layer_names[i]][0].data
	for j in range(layer_nfilters[i]):
		plt.subplot(5,10,j+1)
		im = filters[j,:,:,:]
		im -= im.min()
		im /= (im.max()+0.001)
		im = swap_channel_index(im)
		print im.shape
		plt.imshow(im, interpolation='none')#, cmap = cm.Greys_r)
		plt.axis('off')
	plt.savefig(layer_names[i]+'_23000')

        plt.clf()
	filter_norms = []
	for j in range(layer_nfilters[i]):
		im = filters[j,0,:,:]
		filter_norms.append(np.linalg.norm(im))
	filter_norms.sort()
	plt.plot(filter_norms)
	plt.savefig ("weights_" + layer_names[i])

plt.clf()
filters = net.params['ip1'][0].data
filter_norms = []
for j in range(400):
	fil = filters[j,:]
	filter_norms.append(np.linalg.norm(fil))
filter_norms.sort()
plt.plot(filter_norms)
plt.savefig("fully_connected_weights")
exit()

# Evolution
test_step = 400
for j in range(4):#layer_nfilters[0]):
	i = test_step
	aux = 1
        plt.clf()
	while i<=5000:
		PRETRAINED = 'snapshot/hitsnet_snapshot_iter_'+str(i)+'.caffemodel'
		net = caffe.Classifier(MODEL_FILE, PRETRAINED, mean=hits_mean_npy[0,:,:], raw_scale=255)
		filters = net.params['conv1'][0].data
		plt.subplot(1,15,aux)
		aux += 1
		im = filters[j,0,:,:]
		im -= im.min()
		im /= (im.max()+0.001)
		plt.imshow(im, cmap = cm.Greys_r, interpolation='none')
		plt.axis('off')
		i += test_step
	plt.savefig("iterations_" + str(j))
		
		#filter_norms = []
		#for j in range(layer_nfilters[0]):
			#im = filters[j,0,:,:]
			#filter_norms.append(np.linalg.norm(im))
		#filter_norms.sort()
		#plt.plot(filter_norms)
		#plt.show()
# Evolution
test_step = 200
for j in range(layer_nfilters[1]):
	i = test_step
	aux = 1
        plt.clf()
	while i<=5000:
		PRETRAINED = 'snapshot/hitsnet_snapshot_iter_'+str(i)+'.caffemodel'
		net = caffe.Classifier(MODEL_FILE, PRETRAINED, mean=hits_mean_npy[0,:,:], raw_scale=255)
		filters = net.params['conv2'][0].data
		plt.subplot(2,15,aux)
		aux += 1
		im = filters[j,0,:,:]
		im -= im.min()
		im /= (im.max()+0.001)
		plt.imshow(im, cmap = cm.Greys_r, interpolation='none')
		plt.axis('off')
		i += test_step
	plt.savefig ("evolution_" + str(j))
