from .target_specific_files_in_folder import target_specific_files_in_folder
from config.constants import DESIRED_CSV_TEST_FILE

def get_csv_for_participants():
    all_files = target_specific_files_in_folder()
    csv_files = []
    for file in all_files:
        if file['name'].endswith('.csv') and file['name'] == DESIRED_CSV_TEST_FILE:
            csv_files.append(file)
    return csv_files

def main():
    csv_files = get_csv_for_participants()
    for file in csv_files:
        print(file)

if __name__ == "__main__":
    main()