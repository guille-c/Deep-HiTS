# Deep-HiTS: Rotation Invariant Convolutional Neural Network for Transient Identification

Deep-HiTS is a rotation invariant convolutional neural network (CNN) model for classifying images of transients candidates into artifacts or real sources for the High cadence Transient Survey (HiTS). CNNs have the advantage of learning the features automatically from the data while achieving high performance.

## Running the code

The main code is located in the `src/` directory and is run like this:

```
python Deep-HiTS.py input.txt
```
where `input.txt` is a file that gives parameters to the network. An example parameter file will look like this:

```
[vars]
arch_py: arch7.py
path_to_chunks: /home/shared/Fields_12-2015/chunks_feat_5000/
n_cand_chunk: 5000
base_lr: 0.04
gamma: 0.5
stepsize: 100000
momentum: 0.0
n_epochs: 100
batch_size: 50
N_train = 1220000
N_valid = 100000
N_test = 100000
validate_every_batches = 5000
activation_function: PReLU
tiny_train: False
resume: None
savestep: 25000
improvement_threshold: 0.99
ini_patience: 100000
data_interface: directory
```
- `arch_py`: file where the architecture is defined (see below)
- path_to_chunks: /home/shared/Fields_12-2015/chunks_feat_5000/
- n_cand_chunk: 5000
- base_lr: 0.04
- gamma: 0.5
- stepsize: 100000
- momentum: 0.0
- n_epochs: 100
- batch_size: 50
- N_train = 1220000
- N_valid = 100000
- N_test = 100000
- validate_every_batches = 5000
- activation_function: PReLU
- tiny_train: False
- resume: None
- savestep: 25000
- improvement_threshold: 0.99
- ini_patience: 100000
- data_interface: 


##Reading data

Data i
