from .target_specific_files_in_folder import target_specific_files_in_folder

def get_list_of_subjects():
    all_files = target_specific_files_in_folder()
    subjects = set()
    for file in all_files:
        file_name = file['name']
        if 'sub-' in file_name:
            parts = file_name.split('-')
            subjects.add(parts[1])
    
    return sorted(list(subjects))
