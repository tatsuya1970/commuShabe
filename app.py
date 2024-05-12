from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)
model = tf.keras.models.load_model('great_person_quotes_model.h5')

# Tokenizerのロード（事前にトレーニング時のものを保存しておく必要がある）
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    sequence = text_to_sequence(text)
    prediction = model.predict(sequence)
    match_percentage = prediction[0][0] * 100  # パーセンテージ表記に変換
    # テンプレートにテキストと結果の両方を渡す
    return render_template('index.html', text=text, result=f'名言度: {match_percentage:.2f}%')



def text_to_sequence(text):
    # テキストをシーケンスに変換
    sequences = tokenizer.texts_to_sequences([text])
    # パディングを行う（モデルの入力サイズに合わせて調整する）
    padded = pad_sequences(sequences, maxlen=30, padding='post')  # maxlenはモデルの設定に依存
    return padded

if __name__ == '__main__':
    app.run(debug=True)
