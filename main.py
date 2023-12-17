# main.py
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer  # Tambahkan import ini

app = Flask(__name__)
model = tf.keras.models.load_model("sentiment_analysis_model.h5")

# Definisikan tokenizer
tokenizer = Tokenizer()
# Tambahkan langkah-langkah konfigurasi tokenizer yang diperlukan, misalnya:
sequence_dict = tokenizer.word_index

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']

    # Tokenisasi dan pad teks
    tokenizer.fit_on_texts([text])
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=20)

    # Melakukan prediksi menggunakan model
    prediction = model.predict(padded_sequence)
    sentiment = int(tf.argmax(prediction, axis=1))

    # Mengubah hasil prediksi menjadi kategori sentimen
    if sentiment == 0:
        category = 'Negative'
    elif sentiment == 1:
        category = 'Neutral'
    else:
        category = 'Positive'

    # Print hasil prediksi
    print('---')
    print('Input:', text)
    print('Sequence:', sequence)
    print('Padded Sequence:', padded_sequence)
    print('Prediction:', prediction)
    print('Sentiment:', sentiment, '=', category)
    print('---')
    
    # Mengembalikan hasil prediksi dalam bentuk JSON
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)


# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     print(data)
#     prediction = model.predict(data['input'])
#     return jsonify({'prediction': prediction.tolist()})

# if __name__ == '__main__':
#     app.run()

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Menerima data JSON dari permintaan
#         data = request.get_json()

#         # Memastikan bahwa 'input' ada dalam data JSON
#         if 'input' in data:
#             input_text = data['input']

#             # Lakukan prediksi atau pemrosesan sesuai kebutuhan
#             prediction = model.predict(input_text)

#             # Mengembalikan hasil prediksi dalam bentuk JSON
#             return jsonify({'prediction': prediction.tolist()})

#         else:
#             # Jika 'input' tidak ada dalam data JSON, kembalikan pesan error
#             return jsonify({'error': 'Input tidak valid'})

#     except Exception as e:
#         # Menangani error umum dan mengembalikan pesan error
#         return jsonify({'error': str(e)})

# if __name__ == '__main__':
#     # Menjalankan aplikasi Flask
#     app.run()
