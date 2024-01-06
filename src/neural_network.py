import os
os.environ["KERAS_BACKEND"] = "torch"
from generate_train_data import generate_train_test_local, generate_train_test, generate_train_test_local_mjsoul
import keras
from keras_nlp.layers import TransformerEncoder, SinePositionEncoding, TransformerDecoder
from keras.callbacks import Callback
from keras.layers import Input, LSTM, Dropout, Masking, Cropping1D
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam
from LossHistory import LossHistory
import torch
import numpy as np
# from pyserverchan import pyserver


class MyCbk(Callback):
    def __init__(self, model):
        self.model_to_save = model

    def on_epoch_end(self, epoch, logs=None):
        self.model_to_save.save('./model2/tenpai_epoch_%d.model.keras' % epoch)

def acc(y_true, y_pred):
    y_true = torch.tensor(y_true, device=y_pred.device, dtype=torch.float32)
    y_true = torch.nn.functional.normalize(y_true, dim=-1)
    y_pred = torch.nn.functional.normalize(y_pred, dim=-1)
    return torch.sum(y_true * y_pred, axis=-1)


if __name__ == "__main__":
    print(torch.cuda.is_available())
    # x_train, x_test, y_train, y_test = generate_train_test()
    x_train, x_test, y_train, y_test = generate_train_test_local_mjsoul()

    print(x_train.shape)
    print(y_train.shape)
    # triangular_mask_single = np.triu(np.ones((122, 122)), k=0).astype(bool)
    # mask = np.tile(triangular_mask_single, (x_train.shape[0], 1, 1))
    # mask_val = np.tile(triangular_mask_single, (x_test.shape[0], 1, 1))
    # Model
    inp = Input(shape=(x_train[0].shape[0], 52))
    # inp_mask = Input(shape=(x_train.shape[1], x_train.shape[1]))
    x = Masking()(inp)
    x = LSTM(256, return_sequences=True)(x)
    x = Dropout(0.1)(x)
    x = TransformerEncoder(256, 16, dropout=0.1)(x)
    x = Dropout(0.1)(x)
    x = keras.layers.AveragePooling1D(109, data_format="channels_last")(x)
    x = keras.layers.Flatten(data_format="channels_last")(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.1)(x)
    output = Dense(35, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=output)
    # par_model = multi_gpu_model(model, gpus=2)
    par_model = model
    par_model = keras.saving.load_model("model2/tenpai.keras", custom_objects={'acc': acc})
    opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

    # Which kind of loss to use?
    # We should write another metrics
    par_model.compile(#loss='cosine_proximity',
                      loss='categorical_crossentropy',
                     
                      optimizer=opt,
                      metrics=['categorical_crossentropy', acc])
    print(par_model.summary())

    epoch_nb = 10
    batch = 512

    cbk = MyCbk(model)
    history = LossHistory()

    par_model.fit(x_train, y_train, batch_size=batch, epochs=epoch_nb,
                  verbose=1, validation_data=(x_test, y_test), callbacks=[cbk, history])
    history.loss_plot()
    print("training done")
    par_model.save("model2/tenpai_mjsoul.keras")
    # svc = pyserver.ServerChan()
    # svc.output_to_weixin('Tenpai train done.')

