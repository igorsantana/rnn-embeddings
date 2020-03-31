# RNN Embeddings

## Jointly learning music embeddings with Recurrent Neural Networks

This repository contains all the code that I did during my masters @ State University of Maringá. I do not intend to add new features to this project, as I will not continue this project in a PhD. To better understand what is the goal of this project, this quote is from my thesis and summarizes what I did: 

> This work's goal is to use Recurrent Neural Networks to acquire contextual information for each song, given the sequence of songs that each user has listened to using embeddings. 


If you have any doubts about the code, or want to use it in your project, let me know! I will be glad to help you in anything you need.

### Installation and Setup

As this code was written in Python, I highly recommend you to use [conda](https://docs.conda.io/en/latest/) to install all the dependencies that you'll need to run it. I have provided the [environment file](environment.yml) that I ended up with, and to create the repository using this file, you should run the following command (assuming you already have conda):

```
conda env create -f environment.yml
```

It is important to know that I used Tensorflow 1.14.0, Cuda 9.2 and Python 3.6.9 to run the experiments. If you cannot run with the environment file that I have provided, perhaps its because one of those versions.

### Directory Structure and General Instructions

```
.
|-- analysis
|-- configs
|-- dataset
|   |-- dataset #1
|   |-- dataset #2
|   `-- ...
|-- outputs
|-- project
|   |-- data
|   |-- evaluation
|   |-- models
|   `-- recsys
|-- tmp
```

This project follows this directory structure in order to work. The main python files are in the **project** folder, and any change that you'll want to do in the code must be done in the files in this folder. The **outputs** folder will contain the output file for the models that you built.

The **dataset** contains all the datasets that you'll use in the project, and for each dataset, you should create a separate folder for it inside the **dataset** folder. The project will then look for a `listening_history.csv` file inside of this folder to run it. This file **must be** comma-separated. 

A temporary folder, **tmp**, will be created while the project works. For each dataset that you'll run this project with, a folder inside the **tmp** folder will be created. There you can find the cross-validation folds, the models that you built and the individual recommendations for each user, as well as some auxiliary matrixes used in the UserKNN algorithm.

I have also included an **analysis** folder that I used to create some graphs with the results. You just have to point to the `main.py` file in the analysis folder where are the results, and it will show an graphical comparison between the models with all the metrics.

The project will only work if you provide a configuration file to it. In my case, I stored my configuration files in the **configs** folder, but feel free to delete the folder if you don' want it. The configuration file contains the parameters for the models, and I don't recommend deleting any parameter even if you are not going to use it. I've included a [sample configuration](configs/config_sample.yml) file that you can use as guideline for your project.


To run the project, you have to pass the config to the `main.py` as a parameter. 

```
$ python main.py --config=configs/config_sample.yml 
``` 


###### DISCLAIMER: 

The `model` and `bi` parameters in the `models/rnn` configuration object are not working, as I hardcoded it in my project. If you want to change the layer (to a GRU or a Simple RNN), you should do it [directly in the code](project/models/rnn.py#L147).


### What is included in this project?

To better understand the project, I highly recommend you to go check the work that I used as a baseline for my model:

- [link](https://doi.org/10.1007/s10791-017-9317-7) -  Wang, D., Deng, S. & Xu, G. Sequence-based context-aware music recommendation. Information Retrieval Journal (2018)

Their work, *music2vec*, is one of the baselines for my RNN model. The following embeddings are implemented in this project:

- music2vec
- doc2vec - [link](https://cs.stanford.edu/~quocle/paragraph_vector.pdf)
- GloVe - [link](https://nlp.stanford.edu/projects/glove/)

To evaluate these embeddings models, the CARS that are implemented are the ones that were proposed by Wang et. al (M-TN, SM-TN, CSM-TN, CSM-UK). Besides the metrics that were used in the paper, I have included MAP, NDCG@5 and Precision@5 as well. The cutoff of these metrics is not configurable, sorry.




---

If you have any doubts about this project, feel free to contact me!
