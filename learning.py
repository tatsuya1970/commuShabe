# ライブラリのインポート
import pandas as pd
import numpy as np
import pickle
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# CSVファイルの読み込み
file_path = 'quotes.csv'  # CSVファイルのパスを指定

#data = pd.read_csv(file_path, encoding='utf-8', encoding_errors='ignore')
data = pd.read_csv(file_path, encoding='utf-8')

# CSVの内容を確認
print(data.head())

# テキストデータとラベルの前処理
quotes = data['Quote'].tolist()  # 名言の列のデータをリストとして取得
labels = data['Label'].values  # ラベルのデータをNumPy配列として取得
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(quotes)
sequences = tokenizer.texts_to_sequences(quotes)
padded_sequences = pad_sequences(sequences, padding='post')

# データセットの分割
split = int(len(padded_sequences) * 0.8)
train_sequences = padded_sequences[:split]
train_labels = labels[:split]
test_sequences = padded_sequences[split:]
test_labels = labels[split:]

# モデルの定義
model = Sequential([
    Embedding(input_dim=10000, output_dim=16, input_length=train_sequences.shape[1]),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dense(24, activation='relu'),
    Dense(1, activation='sigmoid')
])

# モデルのコンパイル
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# モデルのトレーニング
model.fit(train_sequences, train_labels, epochs=3, validation_data=(test_sequences, test_labels))

# モデルの保存
model.save('great_person_quotes_model.h5')


# Tokenizerの保存
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
