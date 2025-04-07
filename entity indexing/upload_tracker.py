import os
import json
import uuid
from typing import List, Dict, Optional, Callable, Set, Union

class UploadTracker:
    def __init__(self, collection_name: str, log_dir: str = "/upload_logs"):
        self.collection_name = collection_name
        self.log_dir = log_dir
        self.log_file = os.path.join(self.log_dir, f"{collection_name}_uploaded_ids.json")
        self.uploaded_ids = set()
        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        self._load_log()
        
    def _load_log(self):
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    self.uploaded_ids = set(json.load(f))
                print(f"Loaded {len(self.uploaded_ids)} previously uploaded IDs from log")
            except Exception as e:
                print(f"Error loading upload log: {str(e)}")
                self.uploaded_ids = set()
        else:
            print("No existing upload log found, starting fresh")
            self.uploaded_ids = set()
            
    def _save_log(self):
        try:
            with open(self.log_file, 'w') as f:
                json.dump(list(self.uploaded_ids), f)
        except Exception as e:
            print(f"Error saving upload log: {str(e)}")
    
    def generate_uuid_from_uri(self, uri: str) -> str:
        """Generate a UUID based on the URI for consistency"""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, uri))
        
    def is_uploaded(self, uuid_str: str) -> bool:
        return uuid_str in self.uploaded_ids
        
    def mark_as_uploaded(self, uuid_list: List[str]):
        self.uploaded_ids.update(uuid_list)
        self._save_log()
        
    def get_uploaded_count(self) -> int:
        return len(self.uploaded_ids)
    
    def extract_uri(self, doc: Dict) -> str:
        uri = doc.get('uri')
        if isinstance(uri, dict) and 'value' in uri:
            return uri['value']
        return uri
    
    def filter_new_entities(self, documents: List[Dict]) -> List[Dict]:
        new_docs = []
        for doc in documents:
            uri = self.extract_uri(doc)
            uuid_str = self.generate_uuid_from_uri(uri)
            if not self.is_uploaded(uuid_str):
                new_docs.append(doc)
                
        skipped = len(documents) - len(new_docs)
        if skipped > 0:
            print(f"Skipping {skipped} already uploaded entities")
            
        return new_docs

def process_batch(tracker: UploadTracker, batch: List[Dict], process_func: Callable):
    batch_uuids = []
    
    for doc in batch:
        uri = tracker.extract_uri(doc)
        uuid_str = tracker.generate_uuid_from_uri(uri)
        batch_uuids.append(uuid_str)
    
    try:
        process_func(batch)
        tracker.mark_as_uploaded(batch_uuids)
        print(f"Batch processed - Total uploaded: {tracker.get_uploaded_count()}")
        return True
    except Exception as e:
        print(f"Error processing batch: {str(e)}")
        return False
