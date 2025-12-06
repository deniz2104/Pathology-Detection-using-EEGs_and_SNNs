from gather_list_of_subjects import get_list_of_subjects
from get_google_drive_service import get_google_drive_service
from get_target_folder_from_google_drive import get_target_folder_from_google_drive
from constants import SUBJECTS_PREFIX,EEG_FOLDER

def collect_eeg_files():
    eeg_folder_links = {}
    
    target_folder = get_target_folder_from_google_drive()
    if not target_folder:
        print("Target folder not found")
        return eeg_folder_links
    
    main_folder_id = target_folder[0]['id']
    service = get_google_drive_service()
    
    subject_query = f"'{main_folder_id}' in parents and mimeType = 'application/vnd.google-apps.folder'"
    page_token = None
    subject_folders_map = {}
    
    while True:
        subject_results = service.files().list(
            q=subject_query,
            fields="nextPageToken, files(id, name)",
            pageToken=page_token,
            pageSize=500
        ).execute()
        
        for folder in subject_results.get('files', []):
            subject_folders_map[folder['name']] = folder['id']
        
        page_token = subject_results.get('nextPageToken')
        if not page_token:
            break
    
    subjects = get_list_of_subjects()
    
    for subject in subjects:
        subject_folder_name = f"{SUBJECTS_PREFIX}{subject}"
        
        if subject_folder_name not in subject_folders_map:
            print(f"Subject folder not found: {subject_folder_name}")
            continue
        
        subject_folder_id = subject_folders_map[subject_folder_name]
        
        eeg_query = f"'{subject_folder_id}' in parents and name = '{EEG_FOLDER}' and mimeType = 'application/vnd.google-apps.folder'"
        eeg_results = service.files().list(
            q=eeg_query,
            fields="files(id)"
        ).execute()

        eeg_folders = eeg_results.get('files', [])
        if not eeg_folders:
            print(f"EEG folder not found for subject {subject}")
            continue

        eeg_folder = eeg_folders[0]
        eeg_folder_id = eeg_folder['id']
    
        eeg_folder_links[subject] = {
            'folder_id': eeg_folder_id
        }

    return eeg_folder_links


if __name__ == "__main__":
    eeg_folders = collect_eeg_files()
