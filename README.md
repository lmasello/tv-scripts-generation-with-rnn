# TV scripts generation with RNNs
The project consists of generating scripts of a TV series using Recurrent Neural Networks (RNNs) and the [Seinfeld dataset](https://www.kaggle.com/thec03u5/seinfeld-chronicles#scripts.csv) of scripts from nine seasons. The model generates fake dialogues such as the one below based on the learned patterns from the training data.

> george: what are you doing here?

> jerry: what?

> george: well, you know...

> jerry:(to the phone) yeah, yeah, i got to get out of here!

> jerry:(still talking) what?

> jerry: i thought he was in the shower.(george laughs)

To generate the TV scripts, the project encompasses the following steps:
* Data preprocessing
  * Create a look-up table to transform the vocabulary words to ids
  * Tokenize punctuation so that symbols are treated separately from words (e.g., bye! is composed by two ids, one for bye and another for the exclamation mark)
  * Prepare feature and target tensors by batching words into data chunks of a given size
* Build the neural network
  * Implement the RNN using [PyTorch's Module class](http://pytorch.org/docs/master/nn.html#torch.nn.Module) using either a GRU or LSTM
  * Set hyperparameters
    * sequence_length
    * batch_size
    * num_epochs
    * learning_rate for an Adam optimizer
    * vocab_size: the number of unique tokens in the vocabulary
    * output_size: the desired size of the output
    * embedding_dim: the embedding dimension
    * hidden_dim: hidden dimension of the RNN
    * n_layers: the number of layers/cells in the RNN
  * Train the model to achieve a training loss less than 3.5


## Project implementation
The project implementation complies with Udacity's [list of rubric points](https://review.udacity.com/#!/rubrics/2260/view) required to pass the project. The whole implementation can be found in either the [dlnd_tv_script_generation.ipynb](./dlnd_tv_script_generation.ipynb) Jupyter notebook or [dlnd_tv_script_generation.html](./dlnd_tv_script_generation.html) file.

### Model architecture

## Notes
This project contains my implementation of the "Generate TV scripts" project for the [Udacity's Deep Learning program](https://www.udacity.com/course/deep-learning-nanodegree--nd101). The baseline code has been taken from Udacity's Deep Learning [repository](https://github.com/udacity/deep-learning-v2-pytorch).

Although the model generates dialogues between multiple characters that say complete sentences, questions and answers, it is not perfect. Particularly, the model lacks a level of coherence in some instances. This is expected due to the time invested to train the model and size of the dataset.
