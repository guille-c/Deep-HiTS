# Deep-HiTS: Rotation Invariant Convolutional Neural Network for Transient Identification

Deep-HiTS is a rotation invariant convolutional neural network (CNN) model for classifying images of transients candidates into artifacts or real sources for the High cadence Transient Survey (HiTS). CNNs have the advantage of learning the features automatically from the data while achieving high performance.

## Running the code

The main code is located in the `src/` directory and is run like this:

```
python Deep-HiTS.py input.txt
```
where `input.txt` is a file that gives parameters to the network. 

### Input file

An example parameter file will look like this:

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
- `arch_py`: file where the architecture is defined. The architecure definition file for Deep-HiTS is located at `runs/arch7/arch7.py`
- `path_to_chunks`: Path where the data chunks are located
- `n_cand_chunk`: candidates per chunk
- `base_lr`: initial learning rate
- `gamma`: amount the learning rate will be reduced every `stepsize` iterations.
- `stepsize`: we reduce the learning rate every `stepsize` iterations.
- `momentum`: fraction of the previous weight added to the update of the current one.
- `n_epochs`: number of epochs
- `batch_size`: mini-batch size
- `N_train`: Number of instances used for training.
- `N_valid`: Number of instances used for validating.
- `N_test`: Number of instances used for testing.
- `validate_every_batches`: number of batches per calculation of errors on the validation set.
- `activation_function`: activation function. Currently supports `tanh`(hyperbolic tangent), `ReLU`(rectified linear units), and `PReLU` (leaky ReLU).
- `tiny_train`: Always False on the meantime.
- `resume`: prefix of files for resumming the network from a previous model. This is useful in case a previous training crashes, which we can restart from a saved intermediate state. If `None`, network starts training from scratch.
- `savestep`: Number of steps every which the network will be saved.
- `improvement_threshold`: We assumed the model converged when after feeding 100,000 candidates the zero-one loss (fraction of misclassifications) does not go lower than a `improvement_threshold` of the previous loss.
- `ini_patience`: Number of iterations to wait before deciding to stop training.
- `data_interface`: Type of data to read. Can be `directory` or `random``
-- `directory`: inside the directory `path_to_chunks`(see above) we will have three folders: `chunks_train`,  `chunks_validate`, and ``. Chunks of data in each of the is used for training, validating, and testing respectively.
-- `random`: inside the directory `path_to_chunks`(see above) we will have all chunks, and training, validation, and test set will be chosen randomly.

##Data

Data can be downloaded from the [Quimal Archiving System](http://archiving.cmm.uchile.cl/pub/DeepHits/all_chunks.tar.gz).

