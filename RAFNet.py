"""NOTES: Batch data is different each time in keras, which result in slight differences in results."""
import os
import time
import math
import keras
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.regularizers import l2
from keras.models import Input, Model
from scipy.interpolate import splev, splrep
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
from keras.layers import Conv1D, Dense, Dropout, MaxPooling1D,Reshape, multiply, concatenate
from keras.layers import GlobalAveragePooling1D, Dense, Input, LSTM, Bidirectional, Lambda, Permute, RepeatVector
import keras.backend as K

# the path to the dataset
base_dir = "sleep apnea/dataset"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
ir = 3 # interpolate interval
before = 2
after = 2
# normalize
scaler = lambda arr: (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def load_data(path):
    tm = np.arange(0, (before + 1 + after) * 60, step=1 / float(ir))
    with open(os.path.join(base_dir, path), 'rb') as f: # read preprocessing result
        apnea_ecg = pickle.load(f)
    x_train = []
    o_train, y_train = apnea_ecg["o_train"], apnea_ecg["y_train"]
    groups_train = apnea_ecg["groups_train"]
    for i in range(len(o_train)):
        (rri_tm, rri_signal), (ampl_tm, ampl_siganl) = o_train[i]
		# Curve interpolation
        rri_interp_signal = splev(tm, splrep(rri_tm, scaler(rri_signal), k=3), ext=1) 
        ampl_interp_signal = splev(tm, splrep(ampl_tm, scaler(ampl_siganl), k=3), ext=1)
        x_train.append([rri_interp_signal, ampl_interp_signal])
    x_train = np.array(x_train, dtype="float32").transpose((0, 2, 1)) # convert to numpy format
    y_train = np.array(y_train, dtype="float32")
    x_test = []
    o_test, y_test = apnea_ecg["o_test"], apnea_ecg["y_test"]
    groups_test = apnea_ecg["groups_test"]
    for i in range(len(o_test)):
        (rri_tm, rri_signal), (ampl_tm, ampl_siganl) = o_test[i]
		# Curve interpolation
        rri_interp_signal = splev(tm, splrep(rri_tm, scaler(rri_signal), k=3), ext=1)
        ampl_interp_signal = splev(tm, splrep(ampl_tm, scaler(ampl_siganl), k=3), ext=1)
        x_test.append([rri_interp_signal, ampl_interp_signal])
    x_test = np.array(x_test, dtype="float32").transpose((0, 2, 1))
    y_test = np.array(y_test, dtype="float32")
    return x_train, y_train, groups_train, x_test, y_test, groups_test

# learning rate strategy
def step_decay(epoch):
   initial_lrate = 0.001
   drop = 0.5
   epochs_drop = 10.0
   lrate = initial_lrate * math.pow(drop,  
           math.floor((1+epoch)/epochs_drop))
   print("Learning rate: ", lrate)
   return lrate

def CNN_5min(x, weight=1e-3):
    x = Conv1D(16, kernel_size=11, strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
            kernel_regularizer=l2(weight), bias_regularizer=l2(weight))(x)
    x = Conv1D(24, kernel_size=11, strides=2, padding="same", activation="relu", kernel_initializer="he_normal",
            kernel_regularizer=l2(1e-3), bias_regularizer=l2(weight))(x)
    x = MaxPooling1D(pool_size=3,padding="same")(x)
    x = Conv1D(32, kernel_size=11, strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
            kernel_regularizer=l2(1e-3), bias_regularizer=l2(weight))(x)
    x = MaxPooling1D(pool_size=5,padding="same")(x)
    return x

def CNN_1min(x, weight=1e-3):
    x = Conv1D(16, kernel_size=7, strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight))(x)
    x = Conv1D(24, kernel_size=7, strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight))(x)
    x = MaxPooling1D(pool_size=2,padding="same")(x)
    x = Conv1D(32, kernel_size=7, strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight))(x)
    x = MaxPooling1D(pool_size=3,padding="same")(x)
    return x

def create_model(input_shape):
    input_5min = Input(shape=input_shape)
    # Divide the five-minute-long adjacent segments into five one-minute-long segments
    input_1min = Lambda(lambda x: tf.split(x, 5, axis=-2))(input_5min)
    # Extract RRI and Rpeak of adjacent segments
    RRI_Rpeak_5min = Lambda(lambda x: tf.split(x, 2, axis=-1))(input_5min)
    # Extract RRI and Rpeak of target segment
    RRI_Rpeak_1min = Lambda(lambda x: tf.split(x, 2, axis=-1))(input_1min[2])

    RRI_5min = RRI_Rpeak_5min[0] # RRI of adjacent segments
    Rpeak_5min = RRI_Rpeak_5min[1] # Rpeak of adjacent segments
    RRI_1min = RRI_Rpeak_1min[0] # RRI of target segment
    Rpeak_1min = RRI_Rpeak_1min[1] # Rpeak of target segment

    # Target Segment Feature Extractor
    feature_RRI_1min = CNN_1min(RRI_1min)
    feature_Rpeak_1min = CNN_1min(Rpeak_1min)
    feature_1min = concatenate([feature_RRI_1min, feature_Rpeak_1min],axis=-1) 

    # Adjacent Segment Feature Extractor
    feature_RRI_5min = CNN_5min(RRI_5min)
    feature_Rpeak_5min = CNN_5min(Rpeak_5min)    
    feature_5min = concatenate([feature_RRI_5min, feature_Rpeak_5min],axis=-1) 

    # Morphological Attention
    at1 = GlobalAveragePooling1D()(feature_1min)
    at1 = Dense(32,activation='relu')(at1)
    at1 = Dense(64,activation='sigmoid')(at1)
    at1 = Reshape((1, 64))(at1)
    at1 = multiply([feature_5min, at1])

    temporal_feature = Bidirectional(LSTM(32, activation='tanh',return_sequences=True))(at1)

    # Temporal Attention
    at2 = Permute((2, 1))(feature_1min)
    at2 = Dense(30, activation='softmax')(at2)
    at2 = Lambda(lambda x: K.mean(x, axis=1))(at2)
    at2 = RepeatVector(64)(at2)
    at2 = Permute((2, 1))(at2)
    at2 = multiply([temporal_feature, at2])

    # Fusion
    x = concatenate([feature_1min, at2],axis=-1) 

    # Classification
    x = GlobalAveragePooling1D()(x)
    x = Dense(64)(x)
    dp = Dropout(0.5)(x)
    outputs = Dense(2,activation='sigmoid',name="Output_Layer")(dp)
    model = Model(inputs = input_5min, outputs = outputs)
    return model

if __name__ == "__main__":
    path="apnea-ecg.pkl"
    # 5-minute-long signal
    x_train, y_train, groups_train, x_test, y_test, groups_test = load_data(path)
    # Divide the dataset
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=True, stratify=y_train)
    x_test1, y_test1, groups_test1 = x_test, y_test, groups_test
    # onehot
    y_train = keras.utils.to_categorical(y_train, num_classes=2) # Convert to two categories
    y_val = keras.utils.to_categorical(y_val, num_classes=2) # Convert to two categories
    y_test = keras.utils.to_categorical(y_test, num_classes=2) # Convert to two categories
    # creat model
    model = create_model(x_train.shape[1:])
    # compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # checkpoint
    filepath='sleep apnea/weights.best.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    # learning rate
    lr_scheduler = LearningRateScheduler(step_decay)
    callbacks_list = [lr_scheduler, checkpoint]
    history = model.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_val, y_val),callbacks=callbacks_list)

    loss, accuracy = model.evaluate(x_test,y_test) # test the model
    y_score = model.predict(x_test)
    roc=roc_auc_score(y_score=y_score,y_true=y_test)
    output = pd.DataFrame({"y_true": y_test[:, 1], "y_score": y_score[:, 1], "subject": groups_test})
    output.to_csv("sleep apnea/output/RAFNet.csv", index=False)

    filepath='sleep apnea/weights.best.hdf5'
    model =  keras.models.load_model(filepath, custom_objects={'tf': tf})
    y_true, y_pred = y_test1, np.argmax(model.predict(x_test1, batch_size=64, verbose=1), axis=-1)
    C = confusion_matrix(y_true, y_pred, labels=(1, 0))
    TP, TN, FP, FN = C[0, 0], C[1, 1], C[1, 0], C[0, 1]
    acc, sn, sp = 1. * (TP + TN) / (TP + TN + FP + FN), 1. * TP / (TP + FN), 1. * TN / (TN + FP)
    f1=f1_score(y_true, y_pred, average='binary')
    print("testing:")
    print("loss:{}, acc: {}, sn: {}, sp: {}, f1: {}, roc:{}".format(loss, acc, sn, sp, f1, roc))