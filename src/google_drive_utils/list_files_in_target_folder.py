from .get_google_drive_service import get_google_drive_service
from .get_target_folder_from_google_drive import get_target_folder_from_google_drive
from config.constants import FOLDER_NAME

def list_files_in_folder(folder_name=FOLDER_NAME):
    found_folders = get_target_folder_from_google_drive(folder_name)
    service = get_google_drive_service()

    if not found_folders:
        print(f"No folder found with name '{folder_name}'")
        return

    folder_id = found_folders[0]['id']
    print(f"Found folder '{folder_name}' with ID: {folder_id}")

    all_items = []
    page_token = None

    while True:
        results = service.files().list(
            q=f"'{folder_id}' in parents",
            fields="nextPageToken, files(id, name, mimeType)",
            pageSize=470,
            pageToken=page_token
        ).execute()

        items = results.get('files', [])
        all_items.extend(items)

        page_token = results.get('nextPageToken', None)
        if not page_token:
            break

    return all_items
