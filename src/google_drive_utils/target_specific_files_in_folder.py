from .list_files_in_target_folder import list_files_in_folder
from src.config.constants import FOLDER_NAME, FOLDER_TO_KEEP, FORMATS_TO_KEEP, SUBJECTS_PREFIX

def target_specific_files_in_folder(folder_name=FOLDER_NAME):
    files = list_files_in_folder(folder_name)
    accepted_files = []
    
    for file in files:
        file_name = file['name']
        if FOLDER_TO_KEEP in file_name:
            accepted_files.append(file)
        elif any(file_name.endswith(ext) for ext in FORMATS_TO_KEEP):
            accepted_files.append(file)
        elif SUBJECTS_PREFIX in file_name:
            accepted_files.append(file)
    
    return accepted_files
