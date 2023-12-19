# main.py
# Import library yang dibutuhkan
from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

# Inisialisasi Flask
app = Flask(__name__)

# Load model yang sudah di-train
model = load_model("sentiment_analysis_model.h5")

# Definisikan tokenizer
with open('tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Definisikan fungsi untuk melakukan prediksi
@app.route('/predict', methods=['POST'])
def predict():
    # Menerima data JSON dari permintaan
    data = request.json
    text = data['text']

    # Tokenisasi dan pad teks
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=20)

    # Melakukan prediksi menggunakan model
    prediction = model.predict(padded_sequence)
    sentiment = np.argmax(prediction, axis=1)

    # Mengubah hasil prediksi menjadi kategori sentimen
    if sentiment == 0:
        category = 'Negative'
    elif sentiment == 1:
        category = 'Neutral'
    else:
        category = 'Positive'

    # Print hasil prediksi
    # print('---')
    # print('Input:', text)
    # print('Sequence:', sequence)
    # print('Padded Sequence:', padded_sequence)
    # print('Prediction:', prediction)
    # print('Sentiment:', sentiment, '=', category)
    # print('---')
    
    # Mengembalikan hasil prediksi dalam bentuk JSON
    return jsonify({
        'text': text,
        'sequence': sequence,
        'padded_sequence': padded_sequence.tolist(),
        'prediction': prediction.tolist(),
        'sentiment': sentiment.tolist(),
        'category': category
        })

# Menjalankan aplikasi Flask
if __name__ == '__main__':
    app.run(debug=False)


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
