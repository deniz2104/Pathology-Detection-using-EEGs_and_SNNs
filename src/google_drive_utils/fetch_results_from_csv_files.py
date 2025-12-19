from .get_csv_for_participants import get_csv_for_participants
from .get_google_drive_service import get_google_drive_service
import pandas as pd
import io
import time

def get_important_columns_from_dataframe(df):
    return df[['participant_id','p_factor','attention','internalizing','externalizing']].dropna()

def make_subject_id_uppercase(df):
    df['participant_id'] = df['participant_id'].astype(str).str.upper()
    return df

def fetch_content_from_csv_files(max_retries=3, retry_delay=5):
    service = get_google_drive_service()
    csv_files = get_csv_for_participants()

    if not csv_files:
        print("Error: No CSV files found in Google Drive")
        return None

    latest_modifie_csv_file = sorted(csv_files, key  = lambda x: x['modifiedTime'], reverse=True)[0]
    
    print(f"Fetching data from: {latest_modifie_csv_file['name']}")
    
    for attempt in range(max_retries):
        try:
            request = service.files().get_media(fileId=latest_modifie_csv_file['id'])
            file_content = request.execute()
                
            csv_string = file_content.decode('utf-8')
            df = pd.read_csv(io.StringIO(csv_string))
            df = get_important_columns_from_dataframe(df)
            df = make_subject_id_uppercase(df)
            
            print(f"Successfully loaded {len(df)} participants from CSV")
            return df
            
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} - Error fetching {latest_modifie_csv_file['name']}: {e}")
            
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Failed to fetch data after {max_retries} attempts")
                raise

    return None
