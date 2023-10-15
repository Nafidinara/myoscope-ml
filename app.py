import librosa
import numpy as np
import pandas as pd
import pywt
import os
import noisereduce as nr
from scipy.io import wavfile
import uuid
from sklearn import preprocessing
import joblib
from joblib import dump, load
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import entropy
import flask
import json
from flask import Flask, request

app = Flask(__name__)

FILE_PATH = "files/"

def load_model(path):
    return load(path)

def load_wav_file(file_path):
    signal, sr = librosa.load(file_path)
    return signal, sr

def save_reduced_noise(signal, sr):
    reduced_noise = nr.reduce_noise(y = signal, sr=sr, n_std_thresh_stationary=1.5,stationary=True)
    name_file = f'{uuid.uuid4()}.wav'
    wav_file = wavfile.write(FILE_PATH+name_file, sr, reduced_noise)
    return name_file

def mfcc_extraction(file_path):
    x,sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfcc = librosa.feature.mfcc(y=x, sr = sample_rate)
    return mfcc

def shannon_energy_count(file_path):
    x,sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    x_series = pd.Series(x)
    counts = x_series.value_counts()
    entropy_shanon = entropy(counts)
    return entropy_shanon

def wavelet_extraction(data, coeff, dwt, db, level):
    N = np.array(data).size
    a, ds = dwt[0], list(reversed(dwt[1:]))

    if coeff =='a':
        return pywt.upcoef('a', a, db, level=level)[:N]
    elif coeff == 'd':
        return pywt.upcoef('d', ds[level-1], db, level=level)[:N]
    else:
        raise ValueError("Invalid coefficients: {}".format(coeff))

def extract_feature(name_file):
    
    processed_wav = FILE_PATH+name_file
    raw_feature = mfcc_extraction(processed_wav)

    mfcc_mean = np.mean(raw_feature,dtype=np.float64)
    mfcc_std = np.std(raw_feature)
    mfcc_max = np.max(raw_feature)
    mfcc_min = np.min(raw_feature)
    mfcc_med = np.median(raw_feature)
    mfcc_var = np.var(raw_feature)
    mfcc_skew = skew(raw_feature, axis=0, bias=True)
    mfcc_skew_mean = np.mean(mfcc_skew)
    mfcc_Q1 = np.percentile(raw_feature, 25)
    mfcc_Q3 = np.percentile(raw_feature, 75)
    mfcc_IQR = mfcc_Q3 - mfcc_Q1
    mfcc_range = mfcc_max - mfcc_min
    mfcc_kurt = kurtosis(raw_feature, axis=0, bias=True)
    mfcc_kurt_mean = np.mean(mfcc_kurt)
    entropy_raw = shannon_energy_count(processed_wav)

    db = 'db2'
    level = 4
    
    data_wave, sr = librosa.load(processed_wav, res_type='kaiser_fast')
    coeffs = pywt.wavedec(data_wave, db, level=level)
    
    A4 = wavelet_extraction(data_wave, 'a', coeffs, db, level)
    D4 = wavelet_extraction(data_wave, 'd', coeffs, db, level)
    D3 = wavelet_extraction(data_wave, 'd', coeffs, db, 3)
    D2 = wavelet_extraction(data_wave, 'd', coeffs, db, 2)
    D1 = wavelet_extraction(data_wave, 'd', coeffs, db, 1)
    wavelets = A4 + D4 + D3 + D2 + D1

    wavelet_mean = np.mean(wavelets,dtype=np.float64)
    wavelet_std = np.std(wavelets)
    wavelet_max = np.max(wavelets)
    wavelet_min = np.min(wavelets)
    wavelet_med = np.median(wavelets)
    wavelet_var = np.var(wavelets)
    wavelet_skew = skew(wavelets, axis=0, bias=True)
    wavelet_skew_mean = np.mean(wavelet_skew)
    wavelet_Q1 = np.percentile(wavelets, 25)
    wavelet_Q3 = np.percentile(wavelets, 75)
    wavelet_IQR = wavelet_Q3 - wavelet_Q1
    wavelet_range = wavelet_max - wavelet_min
    wavelet_kurt = kurtosis(wavelets, axis=0, bias=True)
    wavelet_kurt_mean = np.mean(wavelet_kurt)

    df_extracted_feature = pd.DataFrame()
    
    df_extracted_feature['MFCC Means'] = [mfcc_mean]
    df_extracted_feature['MFCC std'] = [mfcc_std]
    df_extracted_feature['MFCC max'] = [mfcc_max]
    df_extracted_feature['MFCC min'] = [mfcc_min]
    df_extracted_feature['Entropy'] = [entropy_raw]
    df_extracted_feature['Wavelet Means'] = [wavelet_mean]
    df_extracted_feature['Wavelet std'] = [wavelet_std]
    df_extracted_feature['Wavelet max'] = [wavelet_max]
    df_extracted_feature['Wavelet min'] = [wavelet_min]

    df_extracted_feature['Med_mfcc'] = [mfcc_med]
    df_extracted_feature['Var_mfcc'] = [mfcc_var]
    df_extracted_feature['Skew_mfcc'] = [mfcc_skew_mean]
    df_extracted_feature['Q1_mfcc'] = [mfcc_Q1]
    df_extracted_feature['Q3_mfcc'] = [mfcc_Q3]
    df_extracted_feature['IQR_mfcc'] = [mfcc_IQR]
    df_extracted_feature['MinMax_mfcc'] = [mfcc_range]
    df_extracted_feature['Kurt_mfcc'] = [mfcc_kurt_mean]

    df_extracted_feature['Med_wavelet'] = [wavelet_med]
    df_extracted_feature['Var_wavelet'] = [wavelet_var]
    df_extracted_feature['Skew_wavelet'] = [wavelet_skew_mean]
    df_extracted_feature['Q1_wavelet'] = [wavelet_Q1]
    df_extracted_feature['Q3_wavelet'] = [wavelet_Q3]
    df_extracted_feature['IQR_wavelet'] = [wavelet_IQR]
    df_extracted_feature['MinMax_wavelet'] = [wavelet_range]
    df_extracted_feature['Kurt_wavelet'] = [wavelet_kurt_mean]
    
    return df_extracted_feature

def predicting(df_extracted_feature, loaded_model, name_file):
    featuresPCA = ['MFCC Means', 'MFCC min', 'Wavelet Means', 'Wavelet std', 'Wavelet max','Var_wavelet','Q1_wavelet','MinMax_wavelet','Kurt_wavelet','Skew_mfcc','Q1_mfcc','Kurt_mfcc']

    features = ['MFCC Means', 'MFCC std', 'MFCC max' ,'MFCC min','Entropy', 'Wavelet Means', 'Wavelet std', 'Wavelet max','Wavelet min',
                'Med_mfcc','Var_mfcc','Skew_mfcc','Q1_mfcc','Q3_mfcc','IQR_mfcc','MinMax_mfcc','Kurt_mfcc',
                'Med_wavelet','Var_wavelet','Skew_wavelet','Q1_wavelet','Q3_wavelet','IQR_wavelet','MinMax_wavelet','Kurt_wavelet'
                ]

    x_2 = df_extracted_feature[featuresPCA].values

    scaler_2 =  preprocessing.StandardScaler().fit(x_2)
    X_scaled2 = scaler_2.transform(x_2)

    result = loaded_model.predict(X_scaled2)
    
    if os.path.exists(FILE_PATH+name_file):
        os.remove(FILE_PATH+name_file)
    
    return result

@app.route("/") 
def index(): 
    return "Hello, welcome to myoscope AI model!"

# Define an endpoint for predicting the class of an sound
@app.route('/predict', methods=['POST'])
def predict():
    #load file
    _file = request.files['file']
    signal, sr = load_wav_file(_file)
    name_file = save_reduced_noise(signal, sr)
    extracted_feature = extract_feature(name_file)
    
    loaded_model = load_model('models/stacking_PCA_tuning.pkl')
    predict_result = predicting(extracted_feature, loaded_model, name_file)
    
    return {
        "result": predict_result.tolist()[0]
    }

if __name__ == '__main__':
    app.run(debug=True)
