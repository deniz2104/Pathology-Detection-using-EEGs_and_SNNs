import os
import numpy as np
import matplotlib.pyplot as plt
import mne

class EEGVisualizer:
    @staticmethod
    def print_filter_info(eeg_raw):
        print("\n--- Filter Info ---")
        print("Highpass filter:", eeg_raw.info['highpass'], "Hz")
        print("Lowpass filter:", eeg_raw.info['lowpass'], "Hz")
        print("Line frequency (powerline):", eeg_raw.info['line_freq'], "Hz")
        print("Sampling frequency:", eeg_raw.info['sfreq'], "Hz")
        print("Total samples:", eeg_raw.n_times)
        print("Data shape (channels × samples):", eeg_raw.get_data().shape)
        print("Recording duration (s):", eeg_raw.times[-1])

    @staticmethod
    def print_channel_info(eeg_raw):
        first_ch = eeg_raw.info['chs'][0]
        print(f"Channel '{first_ch['ch_name']}' unit: {first_ch['unit']} (FIFF constant)")
        print(f"Channel '{first_ch['ch_name']}' coil type: {first_ch['coil_type']}")
        print(f"Hardware Range: {first_ch['range']}")
        print(f"Calibration Factor: {first_ch['cal']}")
        print(f"Loc (Physical coordinates): {first_ch['loc']}")

    @staticmethod
    def print_montage_info(eeg_raw):
        montage = eeg_raw.get_montage()
        print(montage)

    @staticmethod
    def print_file_metadata(eeg_raw):
        print("\n--- File Metadata ---")
        print("Filename:", eeg_raw.filenames)
        print("\n--- Data Info ---")
        print("Has data loaded:", eeg_raw.preload)

    @staticmethod
    def print_signal_quality(eeg_raw):
        print("\n--- Signal Quality ---")
        print("Channel types:", eeg_raw.get_channel_types())
        print("Bad channels:", eeg_raw.info['bads'])

    @staticmethod
    def print_annotations(eeg_raw):
        print("\n--- Events ---")
        print("Annotations:", eeg_raw.annotations)
        if len(eeg_raw.annotations) > 0:
            print("\n--- Annotation Details ---")
            print("Descriptions:", eeg_raw.annotations.description)
            print("Onsets (seconds):", eeg_raw.annotations.onset)
            print("Durations:", eeg_raw.annotations.duration)

        print("Internal first sample index:", eeg_raw.first_samp)
        print("Internal last sample index:", eeg_raw.last_samp)

    @staticmethod
    def print_channel_statistics(eeg_raw):
        data = eeg_raw.get_data()
        stds = np.std(data, axis=1)
        dead_channels = np.where(stds == 0)[0]
        living_channels = np.where(stds > 0)[0]

        print(f"Total Channels: {len(stds)}")
        print(f"Dead (Flat) Channels: {len(dead_channels)}")
        print(f"Living (Active) Channels: {len(living_channels)}")

    @staticmethod
    def print_eeg_summary(eeg_raw):
        EEGVisualizer.print_filter_info(eeg_raw)
        EEGVisualizer.print_channel_info(eeg_raw)
        EEGVisualizer.print_montage_info(eeg_raw)
        EEGVisualizer.print_file_metadata(eeg_raw)
        EEGVisualizer.print_signal_quality(eeg_raw)
        EEGVisualizer.print_annotations(eeg_raw)
        EEGVisualizer.print_channel_statistics(eeg_raw)

    @staticmethod
    def plot_power_spectral_density(eeg_raw, duration=4.0):
        sfreq = eeg_raw.info['sfreq']
        n_fft = int(sfreq * duration)

        try:
            spectrum = eeg_raw.compute_psd(n_fft=n_fft)
            spectrum.plot()
        except AttributeError:
            eeg_raw.plot_psd(n_fft=n_fft)

    @staticmethod
    def make_eeg_plots_directory():
        if not os.path.exists("eeg_plots"):
            os.makedirs("eeg_plots")

    @staticmethod
    def plot_sensor_locations(eeg_raw):
        fig = eeg_raw.plot_sensors(kind='3d', show_names=True, title="GSN 129 Sensors")
        EEGVisualizer.make_eeg_plots_directory()
        fig.savefig("eeg_plots/gsn_129_sensors_3d.png", dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_sensor_topomap(eeg_raw):
        fig = eeg_raw.plot_sensors(kind='topomap', show_names=True, title="GSN 129 Topomap")
        EEGVisualizer.make_eeg_plots_directory()
        fig.savefig("eeg_plots/gsn_129_topomap.png", dpi=300, bbox_inches='tight')
        plt.show()
