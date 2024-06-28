import streamlit as st
import joblib
import librosa
import numpy as np

# Function to extract features from a .wav file
def extract_features(file_name):
    y, sr = librosa.load(file_name, sr=None)
    
    # Extract relevant features (similar to training)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    meanfreq = np.mean(spectral_centroid)
    sd = np.std(spectral_centroid)
    median = np.median(spectral_centroid)
    Q25 = np.percentile(spectral_centroid, 25)
    Q75 = np.percentile(spectral_centroid, 75)
    IQR = Q75 - Q25
    
    # Adjust parameters to avoid exceeding the Nyquist frequency
    n_bands = 6  # Number of frequency bands
    fmin = 20.0  # Minimum frequency (set to a small positive number)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=n_bands, fmin=fmin)
    skew = np.mean(spectral_contrast)
    kurt = np.var(spectral_contrast)
    
    sp_ent = np.mean(librosa.feature.spectral_flatness(y=y))
    sfm = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    mode = np.mean(librosa.feature.zero_crossing_rate(y=y))
    centroid = meanfreq  # already calculated above
    harmonic = librosa.effects.harmonic(y)
    meanfun = np.mean(harmonic)
    minfun = np.min(harmonic)
    maxfun = np.max(harmonic)
    
    delta = librosa.feature.delta(y)
    meandom = np.mean(delta)
    mindom = np.min(delta)
    maxdom = np.max(delta)
    dfrange = maxdom - mindom
    modindx = mode  # zero_crossing_rate already calculated above
    
    # Combine all features into an array
    features = [meanfreq, sd, median, Q25, Q75, IQR, skew, kurt, sp_ent, sfm,
                mode, centroid, meanfun, minfun, maxfun, meandom, mindom,
                maxdom, dfrange, modindx]
    
    return np.array(features)

# Load the trained model and scaler
lr_model = joblib.load('C:/Users/akash/OneDrive/Desktop/genderdetection/gender_prediction_model.pkl')
scaler = joblib.load('C:/Users/akash/OneDrive/Desktop/genderdetection/scaler.pkl')

# Streamlit GUI
st.title('Gender Detection from .wav Files')

# Upload .wav file
uploaded_file = st.file_uploader("Choose a .wav file", type="wav")

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    # Extract features and make prediction
    input_features = extract_features(uploaded_file)
    input_features = scaler.transform(input_features.reshape(1, -1))
    predicted_label = lr_model.predict(input_features)

    # Map predicted label to gender
    predicted_gender = "Male" if predicted_label == 1 else "Female"

    st.write("Predicted Gender:", predicted_gender)
