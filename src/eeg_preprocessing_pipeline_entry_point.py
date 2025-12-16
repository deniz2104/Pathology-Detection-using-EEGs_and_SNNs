import os
import multiprocessing
from google_drive_utils.gather_list_of_subjects import get_list_of_subjects
from google_drive_utils.get_google_drive_service import get_google_drive_service
from preprocessing_pipeline.data_loader import EEGDataLoader
from preprocessing_pipeline.eeg_processor import EEGProcessor
from preprocessing_pipeline.feature_extractor import FeatureExtractor
from preprocessing_pipeline.visualization import EEGVisualizer
from preprocessing_pipeline.utils import export_spikes

def process_subject(subject):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    
    data_loader = EEGDataLoader(subject)
    
    try:
        raw_data = data_loader.load_raw_data()
    except Exception as e:
        print(f"Error reading data for {subject}: {e}")
        return

    for eeg_entry in raw_data:
        eeg_raw = eeg_entry['raw']        
        try:
            EEGVisualizer.print_filter_info(eeg_raw)
            eeg_raw = EEGProcessor.fix_scaling_units(eeg_raw)
            eeg_raw = EEGProcessor.make_montage(eeg_raw)

            events, event_id_map = EEGProcessor.extract_events(eeg_raw)

            eeg_raw, events = eeg_raw.resample(eeg_raw.info['sfreq'] // 2, events=events, n_jobs=1)

            eeg_raw = EEGProcessor.apply_bandpass_filter(eeg_raw, n_jobs=1)
            eeg_raw = EEGProcessor.detect_bads_pyprep(eeg_raw)
            
            eeg_raw = EEGProcessor.interpolate_bad_channels(eeg_raw)
            eeg_raw = EEGProcessor.apply_notch_filter(eeg_raw, n_jobs=1)
            
            eeg_raw = EEGProcessor.sanitize_channel_names(eeg_raw)
            eeg_raw = EEGProcessor.set_eeg_reference(eeg_raw)
            EEGProcessor.detect_automatically_artifacts_with_ica(eeg_raw)

            epochs = EEGProcessor.create_epochs(
                eeg_raw, 
                events, 
                event_id=event_id_map
            )

            epochs = EEGProcessor.normalize_epochs_for_snn(epochs)
            features = FeatureExtractor.select_important_frequency_bands(epochs)

            if features.shape[0] == 0:
                print(f"Skipping {eeg_entry['filename']}: No features extracted.")
                continue

            features_scaled = FeatureExtractor.min_max_scale_features(features)
            spikes = FeatureExtractor.encode_to_spikes_poisson(features_scaled)

            export_spikes(subject, eeg_entry['filename'], spikes)

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
