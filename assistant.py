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

def speak(audio):
    engine = pyttsx3.init()
    engine.setProperty('rate', 200)
    engine.setProperty('volume', 1)
    engine.say(audio)
    engine.runAndWait()

def listen_microphone():
    microphone = sr.Recognizer()
    with sr.Microphone() as source:
        microphone.adjust_for_ambient_noise(source, duration=0.8)
        print("Ouvindo:")
        audio = microphone.listen(source)
        with open('recordings/speech.wav', 'wb') as f:
            f.write(audio.get_wav_data())
    try:
        frase = microphone.recognize_google(audio, language='pt-BR')
        print(frase)
    except sr.UnknownValueError:
        frase = ''
        print("Não entendi")
    return frase

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

        if plot:
            plt.figure(figsize=(len(splitted_audio_data), 5))
            plt.barh(loaded_model[1], predictions[0])
            plt.tight_layout()
            plt.show()

        predictions = predictions.argmax(axis = 1)
        predictions = predictions.astype(int).flatten()
        predictions = loaded_model[1][predictions[0]]
        results.append(predictions)

    count_results = [[results.count(x), x] for x in set(results)]

    return max(count_results)

# emotion = predict_sound('triste.wav', loaded_model[2], plot=False)

def play_music_youtube(emotion):
    play=False
    if emotion == 'triste' or emotion == 'medo':
        speak('Você está triste')
        play=True
    if emotion == 'nervosa' or emotion == 'surpreso':
        speak('Você está nervoso')
        play=True
    return play

# play_music_youtube(emotion[1])

def validate_models():
    audio_source = './recordings/speech.wav'
    predictions = predict_sound(audio_source, loaded_model[2], plot=True)
    print(predictions)
listen_microphone()
validate_models()
