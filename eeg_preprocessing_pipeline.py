import os
import pandas as pd
import mne
import tempfile
import numpy as np

from io import BytesIO
from googleapiclient.http import MediaIoBaseDownload
from collect_eeg_data_for_each_subject import collect_eeg_files
from gather_list_of_subjects import get_list_of_subjects
from get_google_drive_service import get_google_drive_service
import constants
import matplotlib.pyplot as plt

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
            fh = BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            
            print(f"Streaming {filename} from Google Drive...")
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    print(f"  Downloaded {int(status.progress() * 100)}%")
            
            fh.seek(0)
            
            print(f"Reading EEG data from {filename}...")
            with tempfile.NamedTemporaryFile(suffix='.set', delete=False) as tmp:
                tmp.write(fh.getvalue())
                tmp_path = tmp.name
            
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

        