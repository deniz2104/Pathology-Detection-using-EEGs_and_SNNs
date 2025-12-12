import numpy as np
import mne
from mne_icalabel import label_components
from pyprep.find_noisy_channels import NoisyChannels
from sklearn.preprocessing import StandardScaler
from config import constants

class EEGProcessor:
    @staticmethod
    def fix_scaling_units(eeg_raw):
        data = eeg_raw.get_data()
        
        max_val = np.max(np.abs(data))
        
        if max_val > 1.0: 
            print(f"Values are too large for Volts.")
            print(f"Scaling by 1e-6...")
            eeg_raw.apply_function(lambda x: x * 1e-6)
            
        elif max_val > 0.01 and max_val <= 1.0:
            print(f"Values look like millivolts (mV).")
            print(f"Scaling by 1e-3...")
            eeg_raw.apply_function(lambda x: x * 1e-3)
            
        else:
            print("Scaling looks correct")
            
        return eeg_raw

    @staticmethod
    def make_montage(eeg_raw):
        montage = mne.channels.make_standard_montage('GSN-HydroCel-129')
        eeg_raw.set_montage(montage, match_case=False)
        return eeg_raw

    @staticmethod
    def apply_bandpass_filter(eeg_raw, l_freq=constants.LOW_PASS_FILTER_HZ, h_freq=constants.HIGH_PASS_FILTER_HZ, n_jobs=1):
        if not eeg_raw.preload:
            eeg_raw.load_data()
        eeg_raw.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=n_jobs)
        return eeg_raw

    @staticmethod
    def detect_bads_pyprep(eeg_raw):
        try:
            nd = NoisyChannels(eeg_raw)
            nd.find_all_bads(ransac=True)
            eeg_raw.info['bads'] = nd.get_bads()
            if 'Cz' in eeg_raw.info['bads']:
                eeg_raw.info['bads'].remove('Cz')
        except ValueError as e:
            print(f"Warning: PyPrep failed (likely data too short): {e}")
        except Exception as e:
            print(f"Warning: PyPrep failed with unexpected error: {e}")

        return eeg_raw

    @staticmethod
    def interpolate_bad_channels(eeg_raw):
        eeg_raw.interpolate_bads(reset_bads=True,method='spline')
        return eeg_raw
    
    @staticmethod
    def apply_notch_filter(eeg_raw, n_jobs=1):
        eeg_raw.notch_filter(freqs=constants.NOTCH_FILTER_FREQUENCIES, picks = 'eeg', method='fir', filter_length='auto', phase='zero', n_jobs=n_jobs)
        return eeg_raw

    @staticmethod
    def sanitize_channel_names(eeg_raw):
        if eeg_raw.info['bads']:
            eeg_raw.info['bads'] = [str(ch) for ch in eeg_raw.info['bads']]
            
        return eeg_raw

    @staticmethod
    def set_eeg_reference(eeg_raw):
        eeg_raw.set_eeg_reference('average', projection=False)
        return eeg_raw

    @staticmethod
    def detect_automatically_artifacts_with_ica(eeg_raw):
        if not eeg_raw.preload:
            eeg_raw.load_data()
        
        eeg_filtered = EEGProcessor.apply_bandpass_filter(eeg_raw.copy())

        try:
            ica_obj = mne.preprocessing.ICA(n_components=20, method='infomax', max_iter="auto", random_state=constants.RANDOM_SEED, fit_params=dict(extended=True)).fit(eeg_filtered)
            ic_labels = label_components(eeg_filtered, ica_obj, method="iclabel")
        
            labels = ic_labels["labels"]

            exclude_categories = ['eye', 'muscle artifact', 'heart beat', 'line noise', 'channel noise']
            exclude_indices = [i for i, label in enumerate(labels) if label in exclude_categories]
            ica_obj.exclude = exclude_indices
            ica_obj.apply(eeg_raw)

        except Exception as e:
            print(f"Warning: ICA failed (likely low rank or short data): {e}")

        return eeg_raw

    @staticmethod
    def extract_events(eeg_raw):
        events, event_id_map = mne.events_from_annotations(eeg_raw)
        return events, event_id_map

    @staticmethod
    def create_epochs(eeg_raw, events, event_id, tmin=0, tmax=4.0, baseline=(0, 0)):
        epochs = mne.Epochs(eeg_raw, events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=baseline, preload=True, reject = None, event_repeated='drop')
        return epochs
    
    @staticmethod
    def normalize_epochs_for_snn(epochs):
        data = epochs.get_data()
        
        for i in range(len(data)):
            scaler = StandardScaler()
            data[i] = scaler.fit_transform(data[i].T).T
            
        epochs._data = data
        return epochs
