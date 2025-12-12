import numpy as np
from mne.time_frequency import psd_array_welch
from sklearn.preprocessing import MinMaxScaler
from config import constants

class FeatureExtractor:
    @staticmethod
    def select_important_frequency_bands(epochs):
        sfreq = epochs.info['sfreq']
        data = epochs.get_data()

        bands = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 12),
            'Beta': (12, 30),
            'Gamma': (30, 100)
            }
        
        n_epochs, n_channels, _ = data.shape
        n_bands = len(bands)
        features = np.zeros((n_epochs, n_channels, n_bands))

        for epoch in range(n_epochs):
            for channel in range(n_channels):
                psd, freqs = psd_array_welch(data[epoch, channel], sfreq=sfreq, fmin=min(band[0] for band in bands.values()), fmax=max(band[1] for band in bands.values()), n_fft=512)
                for j, (_, (fmin, fmax)) in enumerate(bands.items()):
                    idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
                    features[epoch, channel, j] = np.mean(psd[idx_band])

        return features
    
    @staticmethod
    def min_max_scale_features(features):
        n_epochs, n_channels, n_bands = features.shape
        
        if n_epochs == 0:
            return features

        scaler = MinMaxScaler(feature_range=(0, 1))
        
        features_reshaped = features.reshape(-1, n_bands)
        features_scaled = scaler.fit_transform(features_reshaped)
        
        return features_scaled.reshape(n_epochs, n_channels, n_bands)
    
    @staticmethod
    def encode_to_spikes_poisson(features, time_steps=50):
        n_epochs, _, _ = features.shape
        
        flat_features = features.reshape(n_epochs, -1) 
        
        rand_tensor = np.random.rand(time_steps, n_epochs, flat_features.shape[1])
        
        spikes = (rand_tensor < flat_features).astype(float)
        
        return spikes
