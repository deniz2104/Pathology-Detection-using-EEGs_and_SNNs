import os
import pandas as pd
import mne
import matplotlib.pyplot as plt
import tempfile
import numpy as np

from googleapiclient.http import MediaIoBaseDownload
from collect_eeg_data_for_each_subject import collect_eeg_files
from gather_list_of_subjects import get_list_of_subjects
from get_google_drive_service import get_google_drive_service
import constants
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
            
            with tempfile.NamedTemporaryFile(suffix='.set', delete=False) as tmp:
                downloader = MediaIoBaseDownload(tmp, request)
                
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    if status:
                        print(f"  Downloaded {int(status.progress() * 100)}%")
                
                tmp_path = tmp.name

            print(f"Reading EEG data from {filename}...")
            try:
                raw = mne.io.read_raw_eeglab(tmp_path, preload=True)
                return raw
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            
        except Exception as e:
            print(f"✗ Error preprocessing {filename}: {str(e)}")
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

    def make_montage(self, eeg_raw):
        montage = mne.channels.make_standard_montage('GSN-HydroCel-129')
        eeg_raw.set_montage(montage, match_case=False)
        return eeg_raw

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
    
    def apply_bandpass_filter(self, eeg_raw, l_freq=1.0, h_freq=None):
        eeg_raw.filter(l_freq=l_freq, h_freq=h_freq)
        return eeg_raw

    def set_eeg_reference(self, eeg_raw):
        eeg_raw.set_eeg_reference('average', projection=False)
        return eeg_raw

    def interpolate_bad_channels(self, eeg_raw):
        eeg_raw.interpolate_bads(reset_bads=True,method='spline')
        return eeg_raw
    
    def apply_notch_filter(self, eeg_raw):
        eeg_raw.notch_filter(freqs=[60.0,120.0,180.0], picks = 'eeg', method='fir', filter_length='auto', phase='zero')
        return eeg_raw

    def make_eeg_plots_directory(self):
        if not os.path.exists("eeg_plots"):
            os.makedirs("eeg_plots")

    def plot_sensor_locations(self, eeg_raw):
        fig = eeg_raw.plot_sensors(kind='3d', show_names=True, title="GSN 129 Sensors")
        self.make_eeg_plots_directory()
        fig.savefig("eeg_plots/gsn_129_sensors_3d.png", dpi=300, bbox_inches='tight')

    def plot_sensor_topomap(self, eeg_raw):
        fig = eeg_raw.plot_sensors(kind='topomap', show_names=True, title="GSN 129 Topomap")
        self.make_eeg_plots_directory()
        fig.savefig("eeg_plots/gsn_129_topomap.png", dpi=300, bbox_inches='tight')
    
    def select_important_frequency_bands(self, eeg_raw):
        bands = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 12),
            'Beta': (12, 30),
            'Gamma': (30, 100)
            }
        
        eeg_data = {}
        for band_name, (fmin, fmax) in bands.items():
            filtered = eeg_raw.copy().filter(l_freq=fmin, h_freq=fmax)
            eeg_data[band_name] = filtered
        
        return eeg_data
    
    def fix_scaling_units(self, eeg_raw):
        data = eeg_raw.get_data()
        
        max_val = np.max(np.abs(data))
        
        if max_val > 1.0: 
            print(f"DETECTED SCALING ERROR: Values are too large for Volts.")
            print(f"Likely in microvolts (uV). Scaling by 1e-6...")
            eeg_raw.apply_function(lambda x: x * 1e-6)
            print(f"   ✅ New Max: {np.max(np.abs(eeg_raw.get_data())):.5e} V")
            
        elif max_val > 0.01 and max_val <= 1.0:
            print(f"DETECTED SCALING ERROR: Values look like millivolts (mV).")
            print(f"Scaling by 1e-3...")
            eeg_raw.apply_function(lambda x: x * 1e-3)
            print(f"   ✅ New Max: {np.max(np.abs(eeg_raw.get_data())):.5e} V")
            
        else:
            print("✅ Scaling looks correct (Standard EEG amplitude range).")
            
        return eeg_raw
    
    def detect_bads_pyprep(self, eeg_raw):
        filtered_eeg_raw = self.apply_bandpass_filter(eeg_raw.copy())
        nd = NoisyChannels(filtered_eeg_raw)
        nd.find_all_bads(ransac=True)
        eeg_raw.info['bads'] = nd.get_bads()
        if 'Cz' in eeg_raw.info['bads']:
            eeg_raw.info['bads'].remove('Cz')

        return eeg_raw

def main():
    subjects = get_list_of_subjects()
    for subject in subjects[:1]:
        pipeline = EEGPreprocessingPipeline(subject)
        
        raw_data = pipeline.process_raw_eeg_data()
        for eeg_entry in raw_data:
            eeg_raw = eeg_entry['raw']
            print(f"\n=== Summary for {eeg_entry['filename']} ===")            
            eeg_raw = pipeline.fix_scaling_units(eeg_raw)
            eeg_raw = pipeline.make_montage(eeg_raw)
            eeg_raw = pipeline.detect_bads_pyprep(eeg_raw)
            eeg_raw = pipeline.interpolate_bad_channels(eeg_raw)
            eeg_raw = pipeline.apply_notch_filter(eeg_raw)
            eeg_raw = pipeline.sanitize_channel_names(eeg_raw)
            eeg_raw = pipeline.set_eeg_reference(eeg_raw)
            
            pipeline.plot_power_spectral_density(eeg_raw)

    plt.show()

if __name__ == "__main__":
    main()