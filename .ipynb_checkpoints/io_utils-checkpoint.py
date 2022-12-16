import requests
from pathlib import Path
from pprint import pprint
from tqdm import tqdm
from tqdm.utils import CallbackIOWrapper
import os
import json
import urllib

###########################################################################################

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url,out):
    '''
    download a url download link (str) into the out path (str)
    '''
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1) as t:
        urllib.request.urlretrieve(url, filename=out, reporthook=t.update_to)
        
###########################################################################################

def post_dataset(ACCESS_TOKEN, filename, directoryname,metadata=None):
    '''
    Writes a dataset to a new deposition in zenodo
    returns: request object
    
    ACCESS_TOKEN:  (string) token to zenodo account
    filename:      (string) name of file on local computer
    directoryname: (string) name of folder enclosing file
    '''
    headers = {"Content-Type": "application/json"}
    params = {'access_token': ACCESS_TOKEN}
    path = directoryname+'/'+filename    
    r = requests.post(f'https://zenodo.org/api/deposit/depositions',
                params=params,
                json={},
                headers=headers)
    bucket_url = r.json()['links']['bucket']
    file_size = os.stat(filename).st_size
    with open(filename,'rb') as file:
        with tqdm(total=file_size, unit="B", unit_scale=True, unit_divisor=1024) as t:
            wrapped_file = CallbackIOWrapper(t.update, file, "read")
            r = requests.put(
                bucket_url+'/'+filename,
                data=wrapped_file,
                params=params)
        
    r1 = requests.get('https://zenodo.org/api/deposit/depositions',
                      params={'access_token': ACCESS_TOKEN})
    deposition_id = r1.json()[0]['id']
    
    if metadata==None:
        metadata = {"metadata": {
                            "title": f'{filename}',
                            "upload_type": "dataset",
                            "description": "Placeholder",
                            "creators": [{"name": "Zhang, Xinqiao", "affiliation": "Lehigh University"}]
                            }
                        }
    insert_metadata(ACCESS_TOKEN,deposition_id,metadata)
    return r



def insert_dataset(ACCESS_TOKEN,deposition_id, filename):
    '''
    Adds a dataset to a new deposition in zenodo
    returns: request object
    
    ACCESS_TOKEN:  (string) token to my zenodo account
    deposition_id:      (int) name of file only on computer
    filename: (string) name of folder enclosing file
    '''
        
    r = requests.get(f'https://zenodo.org/api/deposit/depositions/{deposition_id}',
                      params={'access_token': ACCESS_TOKEN})
    bucket_url  = r.json()['links']['bucket']
    headers={"Accept":"application/json",
                "Authorization":"Bearer %s" % ACCESS_TOKEN,
                "Content-Type":"application/octet-stream"}
    key = Path(filename).parts[-1]
    
    file_size = os.stat(filename).st_size
    with open(filename,'rb') as file:
        with tqdm(total=file_size, unit="B", unit_scale=True, unit_divisor=1024) as t:
            wrapped_file = CallbackIOWrapper(t.update, file, "read")
            r1 = requests.put(bucket_url+'/'+key,
                            data = wrapped_file,
                            headers = headers, 
                            params = {'access_token': ACCESS_TOKEN})
    
    return r1


def insert_metadata(ACCESS_TOKEN,deposition_id, data):
    '''
    Writes metadata to a new deposition in zenodo
    returns: request object
        
    ACCESS_TOKEN:  (string) token to my zenodo account
    deposition_id:      (int) name of file only on computer
    filename: (string) name of folder enclosing file
    '''

    url = f"https://zenodo.org/api/deposit/depositions/{deposition_id}"
    headers = {"Content-Type": "application/json"}
    params = {'access_token': ACCESS_TOKEN}

    r = requests.put(url, data=json.dumps(data), headers=headers, params=params)
    return r