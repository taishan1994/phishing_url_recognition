import pandas as pd
import numpy as np
import time
from keras_preprocessing import sequence
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
import keras.callbacks as ckbs

from models import Models


TEST_RESULTS = {'data': {},
                "embedding": {},
                "hiperparameter": {},
                "test_result": {}}

class CustomCallBack(ckbs.Callback):

    def __init__(self):
        ckbs.Callback.__init__(self)
        TEST_RESULTS['epoch_times'] = []

    def on_epoch_begin(self, batch, logs=None):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        TEST_RESULTS['epoch_times'].append(time.time() - self.epoch_time_start)

class PhishingUrlDetection:
    def __init__(self):
        self.params = {'loss_function': 'categorical_crossentropy',
                       'optimizer': 'adam',
                       'seq_len': 200,
                       'batch_train': 5000,
                       'batch_test': 10000,
                       'cate': 2,
                       'epoch': 10,
                       'vocab_size':None,
                       'embedding_dimension': 100,
                       'architecture': "cnn",
                       'result_dir': "result/",
                       'dataset_dir': "数据/small_dataset"}
        self.models = Models(self.params['cate'],
                             self.params['embedding_dimension'],
                             self.params['seq_len'],
                             self.params['vocab_size'])

    def load_data(self):
        featureSet = pd.read_csv('数据/small_dataset/feats.csv', index_col=0)
        X = featureSet['url'].values
        y = featureSet['label'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
        X_train = [i for i in X_train]
        X_test = [i for i in X_test]
        y_train = [i for i in y_train]
        y_test = [i for i in y_test]
        tokenizer = Tokenizer(lower=True, char_level=True, oov_token='-n-')
        tokenizer.fit_on_texts(X_train + X_test)
        self.models.vocab_size = len(tokenizer.word_index)
        x_train = np.asanyarray(tokenizer.texts_to_sequences(X_train))
        x_test = np.asanyarray(tokenizer.texts_to_sequences(X_test))

        y_train = np_utils.to_categorical(y_train, num_classes=self.params['cate'])
        y_test = np_utils.to_categorical(y_test, num_classes=self.params['cate'])

        return (x_train, y_train), (x_test, y_test)

    def fit(self, x_train, y_train, x_test, y_test):
        x_train = sequence.pad_sequences(x_train, maxlen=self.params['seq_len'])
        x_test = sequence.pad_sequences(x_test, maxlen=self.params['seq_len'])

        if self.params['architecture'] == "cnn":
            model = self.models.cnn_base()

        model.compile(loss=self.params['loss_function'], optimizer=self.params['optimizer'], metrics=['accuracy'])
        print(model.summary())

        hist = model.fit(x_train, y_train,
                         batch_size=self.params['batch_train'],
                         epochs=self.params['epoch'],
                         shuffle=True,
                         validation_data=(x_test, y_test),
                         callbacks=[CustomCallBack()])

        t = time.time()
        score, acc = model.evaluate(x_test, y_test, batch_size=self.params['batch_test'])
        TEST_RESULTS['test_result']['test_time'] = time.time() - t
        y_test = list(np.argmax(np.asanyarray(np.squeeze(y_test), dtype=int).tolist(), axis=1))
        y_pred = model.predict_classes(x_test, batch_size=self.params['batch_test'], verbose=1).tolist()
        report = classification_report(y_test, y_pred, target_names=['legitimate', 'phishing'])
        print(report)
        TEST_RESULTS['test_result']['report'] = report
        TEST_RESULTS['epoch_history'] = hist.history
        TEST_RESULTS['test_result']['test_acc'] = acc
        TEST_RESULTS['test_result']['test_loss'] = score

        test_confusion_matrix = confusion_matrix(y_test, y_pred)
        TEST_RESULTS['test_result']['test_confusion_matrix'] = test_confusion_matrix.tolist()

        print('Test loss: {0}  |  test accuracy: {1}'.format(score, acc))


if __name__ == '__main__':
    phishingUrlDetection = PhishingUrlDetection()
    (x_train, y_train), (x_test, y_test) = phishingUrlDetection.load_data()
    phishingUrlDetection.fit(x_train, y_train, x_test, y_test)