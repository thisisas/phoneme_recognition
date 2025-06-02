
import numpy as np
from tensorflow.keras.utils import Sequence
import tensorflow as tf

class PaddedBatchGenerator(Sequence):
    """PaddedBatchGenerator
    Class for sequence length normalization.
    Each sequence is normalized to the longest sequence in the batch
    or a specified length by zero padding. 
    """

    debug = False

    def __init__(self, corpus, utterances, batch_size=20,
                 shuffle=False):
        """
        PaddedBatchGenerator
        Generate a tensor of examples from the specified corpus.
        :param corpus:   Corpus object, must support get_features, get_labels
        :param utterances: List of utterances in the corpus, e.g. a list
            of files returned by corpus.get_utterances("train")
        :param batch_size: mini batch size
        :param shuffle:  Shuffle instances each epoch
        """
        # assign parameters to the corresponding attributes of the corpus
        self.corpus = corpus
        self.utterances = utterances
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._epoch = 0
        self.order = np.arange(len(utterances))
        # shuffle order of indices if shuffle is set to true
        if shuffle:
            np.random.shuffle(self.order)
        
    @property
    def epoch(self):
        return self._epoch

    def __len__(self):
        """len() - Number of batches in data"""
        # calculate the number of batches to cover all utterances using ceiling division to make sure all the
        # examples are included
        return int(np.ceil(len(self.utterances) / self.batch_size))

    def __getitem__(self, batch_idx):
        """
        Get idx'th batch
        :param batch_idx: batch number
        :return: (examples, targets) - Returns tuple of data and targets for
           specified
        """
        # get the start index of the batch
        start_idx = batch_idx * self.batch_size
        # get the ending index of the batch
        end_idx = min((batch_idx + 1) * self.batch_size, len(self.utterances))
        # get all the utterances in the current batch
        batch_utterances = [self.utterances[i] for i in self.order[start_idx:end_idx]]
        # get the features for the utterances in the batch
        features = [self.corpus.get_features(i) for i in batch_utterances]
        # initialize list for the labels
        targets = []
        batch_targets = []
        # loop through all the utterances in the batch
        for utt in batch_utterances:
            # get the starting, ending time of the frames and the labels
            start_frame, stop_frame, phones = self.corpus.get_labels(utt)
            # get the duration of the frames
            frame_dur = stop_frame + 1 - start_frame
            # list to store the labels for each frame
            targets_rep = []
            # get the duration of each frame and extend the phonemes to that duration
            for idx, dur in enumerate(frame_dur):
                targets_rep.extend([phones[idx]] * int(dur))
            # Append all the extended labels to the list of targets
            targets.append(targets_rep)
        final_targets = []
        # calculate the maximum time dimension of each batch for zero-padding
        max_len = max(len(seq) for seq in features)

        for t_list in targets:
            # remove the ending labels (silence) from list of labels to match the features' time dimension
            t_list = t_list[:max_len]
            # convert the labels to one hot encoding
            one_hot_targets = self.corpus.phones_to_onehot(t_list)
            # append all the one hot encoded targets to a list
            final_targets.append(one_hot_targets)
        # zero pad the features and labels array to the length of the longest feature (time dimension) in each batch
        padded_features = [np.pad(seq, ((0, max_len - len(seq)), (0, 0)), mode='constant') for seq in features]
        padded_targets = [np.pad(seq, ((0,max_len - len(seq)), (0, 0)), mode='constant') for seq in final_targets]
        # convert the features and labels to tensors
        padded_label_tensor = tf.convert_to_tensor(padded_targets)
        padded_feature_tensor = tf.convert_to_tensor(padded_features)
        # return the features and labels tensor
        return padded_feature_tensor, padded_label_tensor

        # Hints:  Compute features for each item in batch and keep them in a
        # list.  Then determine longest and fill in missing values with zeros
        # (or other Mask value).

    def on_epoch_end(self):
        """
        on_epoch_end - Bookkeeping at the end of the epoch
        :return:
        """
        # Change these if you use different variables, otherwise nothing to do.
        # Rather than shuffling the data, I shuffle an index list called order
        # which you would need to create in the constructor.
        self._epoch += 1  # Note next epoch
        if self.shuffle:
            np.random.shuffle(self.order)  # reshuffle the data

