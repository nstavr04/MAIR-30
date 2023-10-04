#This file contains function to handle the Speech Regocnition

import pyaudio
import wave
import speech_recognition as sr
import warnings
from scipy.io import wavfile
from scipy.signal import wiener

# Record the audio in a .wav file
def record_audio(filename, duration=5, rate=44100):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True, frames_per_buffer=1024)
    frames = []
    for _ in range(0, int(rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

# Remove noise
def preprocess_audio(filename):
    rate, data = wavfile.read(filename)
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            denoised_data = wiener(data)
        except RuntimeWarning:
            print("Warning: Runtime warning caught during denoising. Skipping denoising.")
            return
    wavfile.write(filename, rate, denoised_data.astype('int16'))

# ASR - Google Speech Recognition
def recognize_speech(filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        return text

if __name__ == '__main__':
    filename = 'recorded_audio.wav'
    print("Recording audio...")
    record_audio(filename)
    print("Preprocessing audio...")
    preprocess_audio(filename)
    print("Recognizing speech...")
    recognized_text = recognize_speech(filename)
    print("Recognized Text:", recognized_text)