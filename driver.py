
"""
TIMIT phone recongition experiments
"""
import os

# Add-on modules
from tensorflow.keras.layers import Dense, Dropout, LSTM, Masking, \
    TimeDistributed, BatchNormalization, Conv1D, Input, GRU, MaxPooling1D
from tensorflow.keras import regularizers

import numpy as np

import matplotlib.pyplot as plt

from dsp.features import Features
from timit.corpus import Corpus

from lib.buildmodels import build_model
from myclassifier.crossvalidator import CrossValidator
from myclassifier.recurrent import train_and_evaluate

def main():
    
    adv_ms = 10     # frame advance and length
    len_ms = 20
    
    # Adjust to your system
    # TimitBaseDir = 'C:/Users/corpora/Timit'
    TimitBaseDir = './timit-for-students'
    corpus = Corpus(TimitBaseDir, os.path.join(TimitBaseDir, 'wav'))
    phonemes = corpus.get_phonemes()  # List of phonemes
    phonemesN = len(phonemes)  # Number of categories
    
    # Get utterance keys 
    devel = corpus.get_utterances('train')  # development corpus
    eval = corpus.get_utterances('test')  # evaluation corpus
    
    # Create a feature extractor and tell the corpus to use it.
    features = Features(adv_ms, len_ms, corpus.get_audio_dir())
    corpus.set_feature_extractor(features)
    
    # Example of retrieving features; also allows us to determine
    # the dimensionality of the feature vector
    f = corpus.get_features(devel[0])
    # Determine input shape
    input_dim = f.shape[1]

    # Check if any features have Inf/NaN in them
    data_sanity_check = False
    if data_sanity_check:
        idx = 0
        for utterances in [devel, eval]:
            for u in utterances:
                f = corpus.get_features(u)
                # Check for NaN +/- Inf
                nans = np.argwhere(np.isnan(f))
                infs = np.argwhere(np.isinf(f))
                if len(nans) > 0 or len(infs) > 0:
                    print(u)
                    if len(nans) > 0:
                        print("NaN")
                        print(nans)
                        
                    if len(infs) > 0:
                        print('Inf')
                        print(infs)
                    pass    # Good place for a breakpoint...
                
                idx = idx + 1
                if idx % 100 == 0:
                    print("idx %d"%(idx))
            

    models_rnn = [
        lambda dim, width, dropout, l2 :    
         [(Masking, [], {"mask_value":0., 
                       "input_shape":[None, dim]}),
         (LSTM, [width], { 
             "return_sequences":True,
             "kernel_regularizer":regularizers.l2(l2),
             "recurrent_regularizer":regularizers.l2(l2)
             }),
         (Dropout, [dropout], {}),
         (Dense, [phonemesN], {'activation':'softmax',
                             'kernel_regularizer':regularizers.l2(l2)},
            # The Dense layer is not recurrent, we need to wrap it in
            # a layer that that lets the network handle the fact that
            # our tensors have an additional dimension of time.
            (TimeDistributed, [], {}))
         ],
        # Arch[1] with GRU, a batch normalization layer, and dropout
        lambda dim, width, dropout, l2 :
        [(Masking, [], {"mask_value":0.,
                        "input_shape":[None, dim]}),
         (GRU, [width], {
             "return_sequences":True,
             "kernel_regularizer":regularizers.l2(l2),
             "recurrent_regularizer":regularizers.l2(l2)
         }),
         (BatchNormalization, [], {}),
         (Dropout, [dropout], {}),
         (Dense, [phonemesN], {'activation':'softmax',
                               'kernel_regularizer':regularizers.l2(l2)},
          (TimeDistributed, [], {}))
         ],
        # Arch[2] with LSTM, batch normalization, and dropout
        lambda dim, width, dropout, l2 :
        [(Masking, [], {"mask_value":0.,
                        "input_shape":[None, dim]}),
         (LSTM, [width], {
             "return_sequences":True,
             "kernel_regularizer":regularizers.l2(l2),
             "recurrent_regularizer":regularizers.l2(l2)
         }),
         (BatchNormalization, [], {}),
         (Dropout, [dropout], {}),
         (Dense, [phonemesN], {'activation':'softmax',
                               'kernel_regularizer':regularizers.l2(l2)},
          (TimeDistributed, [], {}))
         ],
        # Arch[3]
        lambda dim, width, dropout, l2 :
        [(Masking, [], {"mask_value":0.,
                        "input_shape":[None, dim]}),
         (Dense, [width], {'activation':'relu',
                           'kernel_regularizer':regularizers.l2(l2)},
          (TimeDistributed, [], {})),
         (BatchNormalization, [], {}),
         (Dropout, [dropout], {}),
         (Dense, [width], {'activation':'relu',
                           'kernel_regularizer':regularizers.l2(l2)},
          (TimeDistributed, [], {})),
         (BatchNormalization, [], {}),
         (Dropout, [dropout], {}),
         (LSTM, [width], {
             "return_sequences":True,
             "kernel_regularizer":regularizers.l2(l2),
             "recurrent_regularizer":regularizers.l2(l2)
         }),
         (Dropout, [dropout], {}),
         (BatchNormalization, [], {}),
         (LSTM, [width], {
             "return_sequences":True,
             "kernel_regularizer":regularizers.l2(l2),
             "recurrent_regularizer":regularizers.l2(l2)
         }),
         (Dropout, [dropout], {}),
         (BatchNormalization, [], {}),
         (Dense, [width], {'activation':'relu',
                           'kernel_regularizer':regularizers.l2(l2)},
          (TimeDistributed, [], {})),
         (BatchNormalization, [], {}),
         (Dropout, [dropout], {}),
         (Dense, [width], {'activation':'relu',
                           'kernel_regularizer':regularizers.l2(l2)},
          (TimeDistributed, [], {})),
         (BatchNormalization, [], {}),
         (Dropout, [dropout], {}),
         (Dense, [phonemesN], {'activation':'softmax',
                               'kernel_regularizer':regularizers.l2(l2)},
          (TimeDistributed, [], {}))
         ],

        # Arch[4]
        lambda dim, width, dropout, l2 :
        [(Masking, [], {"mask_value":0.,
                        "input_shape":[None, dim]}),
         (Dense, [width], {'activation':'relu',
                           'kernel_regularizer':regularizers.l2(l2)},
          (TimeDistributed, [], {})),
         (BatchNormalization, [], {}),
         (Dropout, [dropout], {}),
         (LSTM, [width], {
             "return_sequences":True,
             "kernel_regularizer":regularizers.l2(l2),
             "recurrent_regularizer":regularizers.l2(l2)
         }),
         (Dropout, [dropout], {}),
         (BatchNormalization, [], {}),
         (Dense, [width], {'activation':'relu',
                           'kernel_regularizer':regularizers.l2(l2)},
          (TimeDistributed, [], {})),
         (BatchNormalization, [], {}),
         (Dropout, [dropout], {}),
         (Dense, [phonemesN], {'activation':'softmax',
                               'kernel_regularizer':regularizers.l2(l2)},
          (TimeDistributed, [], {}))
         ],

        #Arch[5]

        lambda dim, width, dropout, l2 :
        [(Masking, [], {"mask_value":0.,
                        "input_shape":[None, dim]}),
         (Dense, [width], {'activation':'relu',
                           'kernel_regularizer':regularizers.l2(l2)},
          (TimeDistributed, [], {})),
         (BatchNormalization, [], {}),
         (Dropout, [dropout], {}),
         (GRU, [width], {
             "return_sequences":True,
             "kernel_regularizer":regularizers.l2(l2),
             "recurrent_regularizer":regularizers.l2(l2)
         }),
         (BatchNormalization, [], {}),
         (Dropout, [dropout], {}),
         (Dense, [width], {'activation':'relu',
                           'kernel_regularizer':regularizers.l2(l2)},
          (TimeDistributed, [], {})),
         (BatchNormalization, [], {}),
         (Dropout, [dropout], {}),
         (Dense, [phonemesN], {'activation':'softmax',
                               'kernel_regularizer':regularizers.l2(l2)},
          (TimeDistributed, [], {}))
         ]

    ]
  
    # Testing different architectures
    model_configs = [
    (0, 90, 0.25, 0.001),
    (0, 50, 0.20, 0.002),
    (1, 120, 0.20, 0.001),
    (1, 60, 0.10, 0.001),
    (2, 120, 0.30, 0.001),
    (2, 90, 0.15, 0.001),
    (3, 90, 0.25, 0.001),
    (4, 90, 0.25, 0.001),
    (5, 90, 0.25, 0.001)
]
    for idx, (model_idx, units, dropout, lr) in enumerate(model_configs):
        print(f"Running model config #{idx+1} - model_idx: {model_idx}, units: {units}, dropout: {dropout}, lr: {lr}")
        rnn = build_model(models_rnn[model_idx](input_dim, units, dropout, lr))


    debug = False
    if debug:
        devel = devel[0:20]
        eval = eval[0:10]

    # Get the error rate, model and loss over training using train_and_evaluate method in recurrent.py
    err, trained_model, loss = train_and_evaluate(corpus, devel, eval, rnn)
    print("breakpoint line to allow us to retain undocked figures")
    breakpoint_line = True


if __name__ == '__main__':
    plt.ion()
    
    main()