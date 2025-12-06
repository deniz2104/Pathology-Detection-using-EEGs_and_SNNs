from get_google_drive_service import get_google_drive_service
from constants import FOLDER_NAME

def get_target_folder_from_google_drive(folder_name = FOLDER_NAME):
    service = get_google_drive_service()

    response = service.files().list(
        q=f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder'",
        fields="files(id, name)"
    ).execute()

    found_folders = response.get('files', [])
    return found_folders