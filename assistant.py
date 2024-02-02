import pyttsx3
import speech_recognition as sr
from playsound import playsound
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import datetime
import webbrowser as wb
import tensorflow as tf
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from modules import commands_responses

commands = commands_responses.commands
responses = commands_responses.responses

sns.set()

hour = datetime.datetime.now().strftime('%H:%M')
date = datetime.date.today().strftime('%d/%B/%Y')
date = date.split('/')

meu_nome = 'Alexio'

chrome_path = '/usr/bin/brave-browser %s'

def search(frase):
    wb.get(chrome_path).open('https://www.google.com/search?q=' + frase)

MODEL_TYPES = ['EMOÇÃO']

def load_model_by_name(model_type):
    if model_type == MODEL_TYPES[0]:
        model = tf.keras.models.load_model('models/speech_emotion_recognition.hdf5')
        model_dict = sorted(list(['neutra', 'calma', 'feliz', 'triste', 'nervosa', 'medo', 'nojo', 'surpreso']))
        SAMPLE_RATE = 48000
    return model, model_dict, SAMPLE_RATE

model_type = 'EMOÇÃO'
loaded_model = load_model_by_name(model_type)

def predict_sound(AUDIO, SAMPLE_RATE, plot = True):
    results = []
    wav_data, sample_rate = librosa.load(AUDIO, sr=SAMPLE_RATE)
    clip, index = librosa.effects.trim(wav_data, top_db=60, frame_length=512, hop_length=64)
    splitted_audio_data = tf.signal.frame(clip, sample_rate, sample_rate, pad_end=True, pad_value=0)
    for i, data in enumerate(splitted_audio_data.numpy()):
        mfcss_features = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
        mfccs_scaled_features = np.mean(mfcss_features.T, axis = 0)
        mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)
        mfccs_scaled_features = mfccs_scaled_features[:,:,np.newaxis]

        predictions = loaded_model[0].predict(mfccs_scaled_features, batch_size=32)
        print(predictions)

predict_sound('triste.wav', loaded_model[2], plot=True)