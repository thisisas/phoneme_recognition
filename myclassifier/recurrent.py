import math
import time
import os.path

import numpy as np

import keras
from keras.callbacks import TensorBoard
from keras.utils import np_utils
from keras import metrics

import keras.backend as K
import tensorflow as tf

from dsp.utils import Timer

from datetime import datetime

from .batchgenerator import PaddedBatchGenerator
from lib.histories import ErrorHistory, LossHistory
from lib.confusion import ConfusionTensorBoard, plot_confusion

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def train_and_evaluate(corpus, train_utt, test_utt, 
                              model, batch_size=20, epochs=15,
                              name="model"):
    """train_and_evaluate__model(corpus, train_utt, test_utt,
            model, batch_size, epochs)
            
    Given:
        corpus - Object for accessing labels and feature data
        train_utt - utterances used for training
        test_utt - utterances used for testing
        model - Keras model
    Optional arguments
        batch_size - size of minibatch
        epochs - # of epochs to compute
        name - model name
        
    Returns error rate, model, and loss history over training
    """
    # create directory for tensorboard
    log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # split the training utterances (80:20 split) into training and validation utterances for training
    train_u, val_u = train_test_split(train_utt, test_size=0.2, random_state=42)
    # get the utterances for testing the model
    test_u = test_utt
    # compile the model using 'Adam' optimizer and 'categorial crossentropy' loss
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # generate the padded tensors for training, validation and testing the model
    train_generator = PaddedBatchGenerator(corpus, train_u, batch_size)
    val_gen = PaddedBatchGenerator(corpus, val_u, batch_size)
    test_generator = PaddedBatchGenerator(corpus, test_u, batch_size)
    # train the model using the training data
    trained_model = model.fit(train_generator, epochs=epochs, batch_size=batch_size,validation_data=val_gen, callbacks=[tensorboard_callback])
    # get the loss of the model during training
    loss = trained_model.history['loss']
    # list to store all the predictions of the model
    all_predictions = []
    # list to store the true labels
    true_labels=[]
    # set the mask value
    mask_val = 0
    # initialize a list to store all the masks
    masks = []
    # predict the labels for each batch
    for batch_idx in range(len(test_generator)):
        # get the features and true labels for each utterance in the batch
        batch_features, batch_labels = test_generator[batch_idx]
        # get the model predictions for each utterance in the batch
        batch_predictions = model.predict_on_batch(batch_features)
        # get the predicted label by finding the max across the second dimension
        predictions = np.argmax(batch_predictions, axis=2)
        # add the predictions for each batch to the list of all predictions
        all_predictions.append(predictions)
        # get the true label
        truth_label = np.argmax(batch_labels, axis=2)
        # add the true label for each batch to the list of all true labels
        true_labels.append(truth_label)
        # get the mask for each features
        mask = np.all(batch_features != mask_val, axis=2)
        # add masks for each batch to the list of all masks
        masks.append(mask)

    # concatenate each list to forms arrays of the predicted labels, true labels and masks
    predict_arr = np.concatenate(all_predictions, axis=1)
    truth_arr = np.concatenate(true_labels, axis=1)
    fin_masks = np.concatenate(masks,axis=1)
    # plot the confusion matrix
    conf_matrix, fig, ax, bx = plot_confusion(predict_arr, truth_arr, corpus.get_phonemes(), masks=fin_masks)
    # set the title for the confusion matrix
    ax.set_title('Arch[5] with GRU,Dense Layers,BatchNormalization,Dropout=0.25,L2=0.001,Width=90', fontsize=6)
    # get the total number of examples from the confusion matrix
    total_samples = np.sum(conf_matrix)
    # get the total number of correct predictions by summing all values along the diagonal of the confusion matrix
    correct_predictions = np.trace(conf_matrix)
    # calculate the error rate
    err = 1.0 - (correct_predictions / total_samples)

    # return the error rate, model and the loss over training
    return err, trained_model, loss
