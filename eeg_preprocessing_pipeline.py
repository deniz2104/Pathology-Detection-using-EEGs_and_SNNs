import os
import multiprocessing
from pyexpat import features
import pandas as pd
import mne
import matplotlib.pyplot as plt
import tempfile
import numpy as np
import constants
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from mne_icalabel import label_components
from mne.time_frequency import psd_array_welch
from googleapiclient.http import MediaIoBaseDownload
from collect_eeg_data_for_each_subject import collect_eeg_files
from gather_list_of_subjects import get_list_of_subjects
from get_google_drive_service import get_google_drive_service
from pyprep.find_noisy_channels import NoisyChannels

class EEGPreprocessingPipeline:
    def __init__(self, subject_folder_id):
        self.subject_folder_id = subject_folder_id

        if not os.path.exists(constants.EEG_FOLDER_LINKS_CSV):
            collect_eeg_files()

    def get_from_subject_folder_id(self):
        df = pd.read_csv(constants.EEG_FOLDER_LINKS_CSV)
        subject_files = df[df["Subject"] == self.subject_folder_id]
        return subject_files["EEG Folder ID"].to_list()
    
    def get_all_files_from_folder_id(self):
        service = get_google_drive_service()
        folder_ids = self.get_from_subject_folder_id()
        if not folder_ids:
            return []
        folder_id = folder_ids[0]
        query = f"'{folder_id}' in parents and trashed=false"
        results = service.files().list(
            q=query,
            fields="files(id, name)"
        ).execute()

        all_files = results.get('files', [])
        return all_files
    
    def get_only_specific_eeg_files(self):
        all_files = self.get_all_files_from_folder_id()
        eeg_files = [f for f in all_files if f['name'].endswith(constants.EEG_FILE_EXTENSION) and any(task in f['name'] for task in constants.ACCEPTED_TASKS)]
        return eeg_files

    def get_raw_eeg_from_a_subject(self, file_id, filename):
        service = get_google_drive_service() 
        
        try:
            request = service.files().get_media(fileId=file_id)
            
            print(f"Streaming {filename} from Google Drive...")
            
            tmp = tempfile.NamedTemporaryFile(suffix='.set', delete=False)
            downloader = MediaIoBaseDownload(tmp, request)
            
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    print(f"  Downloaded {int(status.progress() * 100)}%")
            
            tmp.close()
            tmp_path = tmp.name

            raw = mne.io.read_raw_eeglab(tmp_path, preload=True)
            
            raw._temp_file_path = tmp_path
            
            return raw
            
        except Exception as e:
            print(f"✗ Error preprocessing {filename}: {str(e)}")
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
            return None
        
    def process_raw_eeg_data(self):
        eeg_files = self.get_only_specific_eeg_files()
        raw_data = []
        for file_info in eeg_files:
            raw = self.get_raw_eeg_from_a_subject(file_info['id'], file_info['name'])
            if raw is not None:
                raw_data.append({
                    'filename': file_info['name'],
                    'raw': raw
                })

        return raw_data
    
    def print_filter_info(self, eeg_raw):
        print("\n--- Filter Info ---")
        print("Highpass filter:", eeg_raw.info['highpass'], "Hz")
        print("Lowpass filter:", eeg_raw.info['lowpass'], "Hz")
        print("Line frequency (powerline):", eeg_raw.info['line_freq'], "Hz")
        print("Sampling frequency:", eeg_raw.info['sfreq'], "Hz")
        print("Total samples:", eeg_raw.n_times)
        print("Data shape (channels × samples):", eeg_raw.get_data().shape)
        print("Recording duration (s):", eeg_raw.times[-1])

    def print_channel_info(self, eeg_raw):
        first_ch = eeg_raw.info['chs'][0]
        print(f"Channel '{first_ch['ch_name']}' unit: {first_ch['unit']} (FIFF constant)")
        print(f"Channel '{first_ch['ch_name']}' coil type: {first_ch['coil_type']}")
        print(f"Hardware Range: {first_ch['range']}")
        print(f"Calibration Factor: {first_ch['cal']}")
        print(f"Loc (Physical coordinates): {first_ch['loc']}")

    def print_montage_info(self, eeg_raw):
        montage = eeg_raw.get_montage()
        print(montage)

    def print_file_metadata(self, eeg_raw):
        print("\n--- File Metadata ---")
        print("Filename:", eeg_raw.filenames)
        print("\n--- Data Info ---")
        print("Has data loaded:", eeg_raw.preload)

    def print_signal_quality(self, eeg_raw):
        print("\n--- Signal Quality ---")
        print("Channel types:", eeg_raw.get_channel_types())
        print("Bad channels:", eeg_raw.info['bads'])

    def print_annotations(self, eeg_raw):
        print("\n--- Events ---")
        print("Annotations:", eeg_raw.annotations)
        if len(eeg_raw.annotations) > 0:
            print("\n--- Annotation Details ---")
            print("Descriptions:", eeg_raw.annotations.description)
            print("Onsets (seconds):", eeg_raw.annotations.onset)
            print("Durations:", eeg_raw.annotations.duration)

        print("Internal first sample index:", eeg_raw.first_samp)
        print("Internal last sample index:", eeg_raw.last_samp)

    def print_channel_statistics(self, eeg_raw):
        data = eeg_raw.get_data()
        stds = np.std(data, axis=1)
        dead_channels = np.where(stds == 0)[0]
        living_channels = np.where(stds > 0)[0]

        print(f"Total Channels: {len(stds)}")
        print(f"Dead (Flat) Channels: {len(dead_channels)}")
        print(f"Living (Active) Channels: {len(living_channels)}")

    def print_eeg_summary(self, eeg_raw):
        self.print_filter_info(eeg_raw)
        self.print_channel_info(eeg_raw)
        self.print_montage_info(eeg_raw)
        self.print_file_metadata(eeg_raw)
        self.print_signal_quality(eeg_raw)
        self.print_annotations(eeg_raw)
        self.print_channel_statistics(eeg_raw)

    def make_montage(self, eeg_raw):
        montage = mne.channels.make_standard_montage('GSN-HydroCel-129')
        eeg_raw.set_montage(montage, match_case=False)
        return eeg_raw

    def plot_power_spectral_density(self, eeg_raw, duration=4.0):
        sfreq = eeg_raw.info['sfreq']
        n_fft = int(sfreq * duration)

        try:
            spectrum = eeg_raw.compute_psd(n_fft=n_fft)
            spectrum.plot()
        except AttributeError:
            eeg_raw.plot_psd(n_fft=n_fft)

    def sanitize_channel_names(self, eeg_raw):
        if eeg_raw.info['bads']:
            eeg_raw.info['bads'] = [str(ch) for ch in eeg_raw.info['bads']]
            
        return eeg_raw
    
    def fix_scaling_units(self, eeg_raw):
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

    def create_epochs(self, eeg_raw, events, event_id, tmin=0, tmax=4.0, baseline=(0, 0)):
        
        epochs = mne.Epochs(eeg_raw, events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=baseline, preload=True, reject = None, event_repeated='drop')
        return epochs
    
    def normalize_epochs_for_snn(self, epochs):
        data = epochs.get_data()
        
        for i in range(len(data)):
            scaler = StandardScaler()
            data[i] = scaler.fit_transform(data[i].T).T
            
        epochs._data = data
        return epochs
    
    def detect_automatically_artifacts_with_ica(self, eeg_raw):
        if not eeg_raw.preload:
            eeg_raw.load_data()
        
        eeg_filtered = self.apply_bandpass_filter(eeg_raw.copy())

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
    
    def apply_bandpass_filter(self, eeg_raw, l_freq=constants.LOW_PASS_FILTER_HZ, h_freq=constants.HIGH_PASS_FILTER_HZ, n_jobs=1):
        if not eeg_raw.preload:
            eeg_raw.load_data()
        eeg_raw.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=n_jobs)
        return eeg_raw

    def set_eeg_reference(self, eeg_raw):
        eeg_raw.set_eeg_reference('average', projection=False)
        return eeg_raw

    def interpolate_bad_channels(self, eeg_raw):
        eeg_raw.interpolate_bads(reset_bads=True,method='spline')
        return eeg_raw
    
    def apply_notch_filter(self, eeg_raw, n_jobs=1):
        eeg_raw.notch_filter(freqs=constants.NOTCH_FILTER_FREQUENCIES, picks = 'eeg', method='fir', filter_length='auto', phase='zero', n_jobs=n_jobs)
        return eeg_raw

    def make_eeg_plots_directory(self):
        if not os.path.exists("eeg_plots"):
            os.makedirs("eeg_plots")

    def plot_sensor_locations(self, eeg_raw):
        fig = eeg_raw.plot_sensors(kind='3d', show_names=True, title="GSN 129 Sensors")
        self.make_eeg_plots_directory()
        fig.savefig("eeg_plots/gsn_129_sensors_3d.png", dpi=300, bbox_inches='tight')
        plt.show()

    def plot_sensor_topomap(self, eeg_raw):
        fig = eeg_raw.plot_sensors(kind='topomap', show_names=True, title="GSN 129 Topomap")
        self.make_eeg_plots_directory()
        fig.savefig("eeg_plots/gsn_129_topomap.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def select_important_frequency_bands(self, epochs):
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
    
    def min_max_scale_features(self, features):
        n_epochs, n_channels, n_bands = features.shape
        
        if n_epochs == 0:
            return features

        scaler = MinMaxScaler(feature_range=(0, 1))
        
        features_reshaped = features.reshape(-1, n_bands)
        features_scaled = scaler.fit_transform(features_reshaped)
        
        return features_scaled.reshape(n_epochs, n_channels, n_bands)
    
    def encode_to_spikes_poisson(self, features, time_steps=50):
        n_epochs, _, _ = features.shape
        
        flat_features = features.reshape(n_epochs, -1) 
        
        rand_tensor = np.random.rand(time_steps, n_epochs, flat_features.shape[1])
        
        spikes = (rand_tensor < flat_features).astype(float)
        
        return spikes

    def export_spikes(self, filename, spikes):
        output_dir = "preprocessed_participants"
        subject_dir = os.path.join(output_dir, f"SUB-{self.subject_folder_id}")
        os.makedirs(subject_dir, exist_ok=True)
        
        task_name = "unknown_task"
        for task in constants.ACCEPTED_TASKS:
            if task in filename:
                task_name = task
                break
        
        output_file = os.path.join(subject_dir, f"{task_name}_spikes.npy")
        np.save(output_file, spikes)
        print(f"Exported spikes for {self.subject_folder_id} - {task_name} to {output_file}")
        
    def extract_events(self, eeg_raw):
        events, event_id_map = mne.events_from_annotations(eeg_raw)
        return events, event_id_map
    
    def detect_bads_pyprep(self, eeg_raw):
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

def process_subject(subject):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    
    pipeline = EEGPreprocessingPipeline(subject)
    
    try:
        raw_data = pipeline.process_raw_eeg_data()
    except Exception as e:
        print(f"Error reading data for {subject}: {e}")
        return

    for eeg_entry in raw_data:
        eeg_raw = eeg_entry['raw']        
        try:
            pipeline.print_filter_info(eeg_raw)
            eeg_raw = pipeline.fix_scaling_units(eeg_raw)
            eeg_raw = pipeline.make_montage(eeg_raw)

            events, event_id_map = pipeline.extract_events(eeg_raw)

            eeg_raw, events = eeg_raw.resample(eeg_raw.info['sfreq'] // 2, events=events, n_jobs=1)

            eeg_raw = pipeline.apply_bandpass_filter(eeg_raw, n_jobs=1)
            eeg_raw = pipeline.detect_bads_pyprep(eeg_raw)
            
            eeg_raw = pipeline.interpolate_bad_channels(eeg_raw)
            eeg_raw = pipeline.apply_notch_filter(eeg_raw, n_jobs=1)
            
            eeg_raw = pipeline.sanitize_channel_names(eeg_raw)
            eeg_raw = pipeline.set_eeg_reference(eeg_raw)
            pipeline.detect_automatically_artifacts_with_ica(eeg_raw)

            epochs = pipeline.create_epochs(
                eeg_raw, 
                events, 
                event_id=event_id_map
            )

            epochs = pipeline.normalize_epochs_for_snn(epochs)
            features = pipeline.select_important_frequency_bands(epochs)

            if features.shape[0] == 0:
                print(f"Skipping {eeg_entry['filename']}: No features extracted.")
                continue

            features_scaled = pipeline.min_max_scale_features(features)
            spikes = pipeline.encode_to_spikes_poisson(features_scaled)

            pipeline.export_spikes(eeg_entry['filename'], spikes)

        finally:
            if hasattr(eeg_raw, '_temp_file_path'):
                if os.path.exists(eeg_raw._temp_file_path):
                    os.remove(eeg_raw._temp_file_path)
                    print(f"Cleaned up temp file: {eeg_raw._temp_file_path}")

def main():
    try:
        get_google_drive_service()
    except Exception as e:
        print(f"Warning: Could not initialize Google Drive service: {e}")

    subjects = get_list_of_subjects()
    
    num_cores = multiprocessing.cpu_count()
    
    with multiprocessing.Pool(processes=num_cores) as pool:
        pool.map(process_subject, subjects)

if __name__ == "__main__":
    main()