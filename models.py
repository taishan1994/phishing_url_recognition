from keras.models import  Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, Flatten, MaxPooling1D
from keras.layers import Embedding, Convolution1D
from keras_self_attention import SeqSelfAttention

class Models:
    def __init__(self, cate, embed_dim, seq_len, vocab_size):
        self.cate = cate
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def rnn_base(self):
        model = Sequential()
        model.add(Embedding(self.vocab_size+1, self.embed_dim))
        model.add(LSTM(128))
        model.add(Dense(self.cate, activation="sigmoid"))
        return model

    def brnn_base(self):
        model = Sequential()
        model.add(Embedding(self.vocab_size + 1, self.embed_dim))
        model.add(Bidirectional(LSTM(128)))
        model.add(Dense(len(self.cate), activation='sigmoid'))
        return model

    def cnn_base(self,):
        model = Sequential()
        model.add(
            Embedding(self.vocab_size + 1, self.embed_dim, input_length=self.seq_len))
        model.add(Convolution1D(128, 3, activation='tanh'))
        model.add(Flatten())

        model.add(Dense(self.cate, activation='sigmoid'))

        return model

    def ann_base(self):
        model = Sequential()

        model.add(
            Embedding(self.vocab_size + 1, self.embed_dim, input_length=self.seq_len))
        model.add(Dense(128, activation='tanh'))
        model.add(Flatten())

        model.add(Dense(self.cate, activation='sigmoid'))

        return model

    def att_base(self):
        model = Sequential()
        model.add(Embedding(self.vocab_size + 1, self.embed_dim, input_length=self.seq_len))
        # model.add(Bidirectional(LSTM(units=128, return_sequences
        model.add(SeqSelfAttention(attention_activation='sigmoid'))
        model.add(Flatten())

        model.add(Dense(self.cate, activation='sigmoid'))

        return model

    def rnn_complex(self):
        model = Sequential()

        model.add(Embedding(self.vocab_size + 1, self.embed_dim))
        model.add(LSTM(128, return_sequences=True))

        model.add(LSTM(128, return_sequences=True))

        model.add(LSTM(128, return_sequences=True))

        model.add(LSTM(128, return_sequences=True))

        model.add(LSTM(128, return_sequences=True))

        model.add(LSTM(128, return_sequences=True))

        model.add(LSTM(128))

        model.add(Dense(self.cate, activation='sigmoid'))

        return model

    def brnn_complex(self):
        model = Sequential()
        model.add(Embedding(self.vocab_size + 1, self.embed_dim))
        model.add(Bidirectional(LSTM(64, return_sequences=True)))

        model.add(Bidirectional(LSTM(64, return_sequences=True)))

        model.add(Bidirectional(LSTM(64, return_sequences=True)))

        model.add(Bidirectional(LSTM(64, return_sequences=True)))

        model.add(Bidirectional(LSTM(64, return_sequences=True)))

        model.add(Bidirectional(LSTM(64, return_sequences=True)))

        model.add(Bidirectional(LSTM(128)))

        model.add(Dense(self.cate, activation='sigmoid'))

        return model

    def ann_complex(self):
        model = Sequential()

        model.add(Embedding(self.vocab_size + 1, self.embed_dim, input_length=self.seq_len))

        model.add(Dense(128, activation='tanh'))

        model.add(Dense(128, activation='tanh'))

        model.add(Dense(128, activation='tanh'))

        model.add(Dense(128, activation='tanh'))

        model.add(Dense(128, activation='tanh'))

        model.add(Dense(128, activation='tanh'))

        model.add(Dense(128, activation='tanh'))
        model.add(Flatten())

        model.add(Dense(self.cate, activation='sigmoid'))

        return model

    def att_complex(self):
        model = Sequential()
        model.add(Embedding(self.vocab_size + 1, self.embed_dim, input_length=self.seq_len))

        model.add(LSTM(units=128, return_sequences=True))

        model.add(Bidirectional(LSTM(units=128, return_sequences=True)))

        model.add(SeqSelfAttention(attention_activation='sigmoid'))

        model.add(Bidirectional(LSTM(units=128, return_sequences=True)))

        model.add(LSTM(units=128, return_sequences=True))

        model.add(SeqSelfAttention(attention_activation='sigmoid'))

        model.add(LSTM(units=128, return_sequences=True))

        model.add(Flatten())

        model.add(Dense(self.cate, activation='sigmoid'))

        return model

    def cnn_complex(self):
        model = Sequential()

        model.add(Embedding(self.vocab_size + 1, self.embed_dim, input_length=self.seq_len))

        model.add(Convolution1D(128, 3, activation='tanh'))
        model.add(MaxPooling1D(3))
        model.add(Dropout(0.2))

        model.add(Convolution1D(128, 7, activation='tanh', padding='same'))
        model.add(Dropout(0.2))

        model.add(Convolution1D(128, 5, activation='tanh', padding='same'))
        model.add(Dropout(0.2))

        model.add(Convolution1D(128, 3, activation='tanh', padding='same'))
        model.add(MaxPooling1D(3))
        model.add(Dropout(0.2))

        model.add(Convolution1D(128, 5, activation='tanh', padding='same'))
        model.add(Dropout(0.2))

        model.add(Convolution1D(128, 3, activation='tanh', padding='same'))
        model.add(MaxPooling1D(3))
        model.add(Dropout(0.2))

        model.add(Convolution1D(128, 3, activation='tanh', padding='same'))
        model.add(MaxPooling1D(3))
        model.add(Dropout(0.2))

        model.add(Flatten())

        model.add(Dense(self.cate, activation='sigmoid'))

        return model

    def cnn_complex2(self):
        model = Sequential()

        model.add(Embedding(self.vocab_size + 1, self.embed_dim, input_length=self.seq_len))

        model.add(Convolution1D(128, 3, activation='tanh'))
        model.add(MaxPooling1D(3))
        model.add(Dropout(0.2))

        model.add(Convolution1D(256, 7, activation='tanh', padding='same'))
        model.add(Dropout(0.2))

        model.add(Convolution1D(96, 5, activation='tanh', padding='same'))
        model.add(Dropout(0.2))

        model.add(Convolution1D(128, 3, activation='tanh', padding='same'))
        model.add(Dropout(0.2))

        model.add(Convolution1D(196, 5, activation='tanh', padding='same'))
        model.add(Dropout(0.2))

        model.add(Convolution1D(128, 3, activation='tanh', padding='same'))
        model.add(Dropout(0.2))

        model.add(Convolution1D(96, 5, activation='tanh', padding='same'))
        model.add(MaxPooling1D(3))
        model.add(Dropout(0.2))

        model.add(Convolution1D(128, 3, activation='tanh', padding='same'))
        model.add(Dropout(0.2))

        model.add(Convolution1D(196, 5, activation='tanh', padding='same'))
        model.add(Dropout(0.2))

        model.add(Convolution1D(128, 7, activation='tanh', padding='same'))
        model.add(Dropout(0.2))

        model.add(Convolution1D(96, 3, activation='tanh', padding='same'))
        model.add(MaxPooling1D(3))
        model.add(Dropout(0.2))

        model.add(Flatten())

        model.add(Dense(self.cate, activation='sigmoid'))

        return model

    def cnn_complex3(self):
        model = Sequential()

        model.add(Embedding(self.vocab_size + 1, self.embed_dim, input_length=self.seq_len))

        model.add(Convolution1D(128, 3, activation='tanh'))
        model.add(MaxPooling1D(3))

        model.add(Convolution1D(256, 7, activation='tanh', padding='same'))
        model.add(Convolution1D(96, 5, activation='tanh', padding='same'))
        model.add(Convolution1D(128, 3, activation='tanh', padding='same'))
        model.add(MaxPooling1D(3))
        model.add(Convolution1D(196, 5, activation='tanh', padding='same'))
        model.add(Convolution1D(128, 3, activation='tanh', padding='same'))
        model.add(Convolution1D(96, 5, activation='tanh', padding='same'))
        model.add(Convolution1D(128, 3, activation='tanh', padding='same'))
        model.add(Convolution1D(196, 5, activation='tanh', padding='same'))
        model.add(Convolution1D(128, 7, activation='tanh', padding='same'))
        model.add(Convolution1D(96, 3, activation='tanh', padding='same'))
        model.add(MaxPooling1D(3))

        model.add(Convolution1D(196, 5, activation='tanh', padding='same'))
        model.add(Convolution1D(128, 7, activation='tanh', padding='same'))
        model.add(MaxPooling1D(3))

        model.add(Convolution1D(196, 5, activation='tanh', padding='same'))
        model.add(Convolution1D(128, 7, activation='tanh', padding='same'))
        model.add(Convolution1D(96, 3, activation='tanh', padding='same'))

        model.add(Flatten())

        model.add(Dense(self.cate, activation='sigmoid'))

        return model

    def custom_model(self):
        model = None

        return model