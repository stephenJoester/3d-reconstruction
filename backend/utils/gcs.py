import os
from google.cloud import storage
from datetime import timedelta

class GCSHandler:
    def __init__(self, bucket_name: str = None, url_expire_minutes: int = 15):
        self.bucket_name = bucket_name or os.getenv("PREDICTION_BUCKET_NAME", "")
        self.url_expire_minutes = url_expire_minutes or int(os.getenv("GCS_FILE_EXPIRE_MINUTES", 15))
        self.client = storage.Client.from_service_account_json("./secrets/key.json")

    def upload_file(self, local_file_path: str, gcs_path: str): 
        """
        Upload a local file to the specified GCS path.
        Returns a signed download URL

        Args:
            local_file_path (str): _description_
            gcs_path (str): _description_
        """
        print("-----Uploading file to GCS...-----")
        bucket = self.client.bucket(self.bucket_name)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_file_path)
        
        url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=self.url_expire_minutes),
            method="GET"
        )
        return url

    def download_file(self, source_blob: str, destination: str):
        """
        Download a file from GCS to a local destination.

        Args:
            source_blob (str): The GCS path of the file to download.
            destination (str): The local path where the file will be saved.
        """
        print("-----Downloading file from GCS...-----")
        bucket = self.client.bucket(self.bucket_name)
        blob = bucket.blob(source_blob)
        blob.download_to_filename(destination)
        print(f"File downloaded to {destination}")
        return destination, blob