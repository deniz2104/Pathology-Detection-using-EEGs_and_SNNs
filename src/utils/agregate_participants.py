import os
from pathlib import Path
from src.google_drive_utils.fetch_results_from_csv_files import fetch_content_from_csv_files

def remove_participants_from_folder_with_no_data(participant_folder: str | None = None):
    if participant_folder is None:
        repo_root = Path(__file__).resolve().parents[2]
        participant_folder = str(repo_root / 'preprocessed_participants')

    df = fetch_content_from_csv_files()
    if df is None or df.empty:
        print("No participant data available to filter.")
        return

    participant_ids_with_data = set(df['participant_id'].astype(str).str.upper().tolist())
    all_folders = os.listdir(participant_folder)

    folders_to_remove = [folder_name for folder_name in all_folders if folder_name not in participant_ids_with_data]

    for folder_name in folders_to_remove:
        folder_path = os.path.join(participant_folder, folder_name)
        if os.path.isdir(folder_path):
            if folder_name not in participant_ids_with_data:
                print(f"Removing folder: {folder_path} as participant ID {folder_name} has no data.")
                try:
                    import shutil
                    shutil.rmtree(folder_path)
                except Exception as e:
                    print(f"Error removing folder {folder_path}: {e}")
