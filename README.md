# Purpose

CloudML aims to provide a set of tools that allow building a classifier on the
cloud. It consists of three components:

1. Trainer, which receives data from the import handler and trains a classifier to produce a classification model,
2. Predictor, that uses a model produced by the trainer to predict the class of incoming requests,
3. Import handler, a utility module that is responsible for feeding the trainer and the predictor with data.
4. Transformer, which receives data from the import handler and trains a transformer.
