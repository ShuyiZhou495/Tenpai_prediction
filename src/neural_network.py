import os
os.environ["KERAS_BACKEND"] = "torch"
from generate_train_data import generate_train_test_local, generate_train_test
import keras
from keras_nlp.layers import TransformerEncoder
from keras.callbacks import Callback
from keras.layers import Input, LSTM, Dropout
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam
from LossHistory import LossHistory
import torch
import tensorflow as tf
# from pyserverchan import pyserver


class MyCbk(Callback):
    def __init__(self, model):
        self.model_to_save = model

    def on_epoch_end(self, epoch, logs=None):
        self.model_to_save.save('./model/tenpai_epoch_%d.model.keras' % epoch)

def acc(y_true, y_pred):
    y_true = torch.tensor(y_true, device=y_pred.device, dtype=torch.float32)
    y_true = torch.nn.functional.normalize(y_true, dim=-1)
    y_pred = torch.nn.functional.normalize(y_pred, dim=-1)
    return torch.sum(y_true * y_pred, axis=-1)


if __name__ == "__main__":
    print(torch.cuda.is_available())
    # x_train, x_test, y_train, y_test = generate_train_test()
    x_train, x_test, x_len, x_len_test, y_train, y_test = generate_train_test_local()
    print(x_train.shape)
    print(y_train.shape)

    # Model
    inp = Input(shape=(x_train[0].shape[0], 52))
    x = Dense(256, activation="relu")(inp)
    # x = keras_nlp.layers.SinePositionEncoding()(x) + x
    x = TransformerEncoder(512, 4, dropout=0.1)(x)
    x = Dropout(0.1)(x)
    # x = keras.layers.AveragePooling1D(109, data_format="channels_last")(x)
    # x = keras.layers.Flatten(data_format="channels_last")(x)
    x = LSTM(256, return_sequences=False)(x)
    x = Dropout(0.1)(x)
    """
    x = CuDNNLSTM(256, return_sequences=True)(x)
    x = Dropout(0.1)(x)
    x = CuDNNLSTM(256, return_sequences=True)(x)
    x = Dropout(0.1)(x)
    """
    

    # x = LSTM(256, return_sequences=True)(inp)
    # x = Dropout(0.1)(x)
    # x = LSTM(256, return_sequences=False)(x)
    # x = Dropout(0.1)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.1)(x)
    output = Dense(34, activation="softmax")(x)

    model = Model(inputs=inp, outputs=output)

    # par_model = multi_gpu_model(model, gpus=2)
    par_model = model

    opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

    # Which kind of loss to use?
    # We should write another metrics
    par_model.compile(#loss='cosine_proximity',
                      loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=[keras.metrics.CategoricalAccuracy()])
    print(par_model.summary())

    epoch_nb = 100
    batch = 512

    cbk = MyCbk(model)
    history = LossHistory()

    par_model.fit(x_train, y_train, batch_size=batch, epochs=epoch_nb,
                  verbose=1, validation_data=(x_test, y_test), callbacks=[cbk, history])
    history.loss_plot()
    print("training done")
    # model.save("../model/tenpai.model")
    # svc = pyserver.ServerChan()
    # svc.output_to_weixin('Tenpai train done.')

