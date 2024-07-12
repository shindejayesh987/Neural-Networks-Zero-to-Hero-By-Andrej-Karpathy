# Neural-Networks-Zero-to-Hero-By-Andrej-Karpathy

Welcome to the repository dedicated to learning deep learning concepts and implementing various NLP models using PyTorch. This repository follows a structured approach from basic neural network training to advanced models like Transformers (inspired by GPT).

## Overview

This repository includes step-by-step tutorials and exercises covering fundamental concepts such as backpropagation, training neural networks, and progressively implementing language models:

- **Bigram Character-Level Language Model:** Introduction to `torch.Tensor`, model training, sampling, and loss evaluation.
- **Exercises:**
  - **E01:** Train a trigram language model and evaluate its loss compared to the bigram model.
  - **E02:** Split the dataset into train, dev, and test sets. Train bigram and trigram models on the train set and evaluate on dev and test sets.
  - **E03:** Tune smoothing strength for the trigram model using the dev set. Evaluate the best setting on the test set.
  - **E04:** Optimize model efficiency by indexing into rows of weight matrices instead of using `F.one_hot`.
  - **E05:** Use `F.cross_entropy` for training and understand its advantages over other methods.
  - **E06:** Meta-exercises for further exploration and innovation in NLP tasks.

- **Multilayer Perceptron (MLP) Language Model:** Implementing an MLP-based character-level language model.
- **Exercises:**
  - **E01:** Tune hyperparameters to achieve better validation loss.
  - **E02:** Investigate the impact of weight initialization strategies on training performance.
  - **E03:** Implement and test ideas from the Bengio et al. (2003) paper on MLP training.

- **Deep Neural Networks (DNNs):** Understanding MLP internals, activation statistics, gradients, and common pitfalls.
- **Exercises:**
  - **E01:** Investigate the effects of initializing all weights and biases to zero on network training.
  - **E02:** Implement batch normalization (BatchNorm) and demonstrate its post-training optimization benefits.

- **Convolutional Neural Networks (CNNs):** Implementing a CNN architecture similar to WaveNet for character-level tasks.
- **Exercises:**
  - **E01:** Explore the training dynamics and optimizations specific to CNNs.
  - **E02:** Understand the benefits of hierarchical deep architectures in sequence modeling.

- **Transformer Models:** Building a Generatively Pretrained Transformer (GPT) based on "Attention is All You Need".
- **Exercises:**
  - **EX1:** Enhance multi-head attention efficiency within the GPT model.
  - **EX2:** Train and fine-tune the GPT model on a custom dataset for practical applications.
  - **EX3:** Pretrain the GPT on a large dataset and finetune on a smaller dataset to observe validation loss improvements.
  - **EX4:** Experiment with additional features from Transformer literature to enhance GPT performance.

- **Tokenization:** Building a Tokenizer for large language models, focusing on encoding and decoding text into tokens.
- **Issues Addressed:** Common problems in LLMs traced back to tokenization and potential solutions.

- **GPT-2 Implementation:** Reproducing GPT-2 (124M) architecture and training process from scratch.
- **Project Setup:** Detailed instructions for setting up environments, installing dependencies, and running each project.

## Jupyter Notebooks Organization

This repository organizes code and exercises into separate Jupyter Notebooks for each video in the playlist. You can explore different topics and
Each directory contains notebooks corresponding to specific videos in the playlist. You can search for and explore topics of interest by navigating to the relevant directory.



## Usage

Clone the repository and navigate to the desired model directory. Follow the README instructions and exercise prompts to explore and learn each topic sequentially.

```sh
git clone (https://github.com/shindejayesh987/Neural-Networks-Zero-to-Hero-By-Andrej-Karpathy)
# Follow setup and usage instructions in README.md
