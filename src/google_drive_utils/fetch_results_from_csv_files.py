from .get_csv_for_participants import get_csv_for_participants
from .get_google_drive_service import get_google_drive_service
import pandas as pd
import io

def fetch_content_from_csv_files():
    service = get_google_drive_service()
    csv_files = get_csv_for_participants()

    latest_modifie_csv_file = sorted(csv_files, key  = lambda x: x['modifiedTime'], reverse=True)[0]
        
    try:
        request = service.files().get_media(fileId=latest_modifie_csv_file['id'])
        file_content = request.execute()
            
        csv_string = file_content.decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_string))
            
    except Exception as e:
        print(f"Error processing {latest_modifie_csv_file['name']}: {e}")

if __name__ == "__main__":
    fetch_content_from_csv_files()
