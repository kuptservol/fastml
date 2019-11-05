from fastml.core.imports import *

def ifnone(a:Any,b:Any)->Any:
    "`a` if `a` is not None, otherwise `b`."
    return b if a is None else a

def download_url(url:str, dest:str, overwrite:bool=False, pbar:ProgressBar=None,
                 show_progress=True, chunk_size=1024*1024, timeout=4, retries=5)->None:
    "Download `url` to `dest` unless it exists and not `overwrite`."
    if os.path.exists(dest) and not overwrite: return

    s = requests.Session()
    s.mount('http://',requests.adapters.HTTPAdapter(max_retries=retries))
    u = s.get(url, stream=True, timeout=timeout)
    try: file_size = int(u.headers["Content-Length"])
    except: show_progress = False

    with open(dest, 'wb') as f:
        nbytes = 0
        if show_progress: pbar = progress_bar(range(file_size), auto_update=False, leave=False, parent=pbar)
        try:
            for chunk in u.iter_content(chunk_size=chunk_size):
                nbytes += len(chunk)
                if show_progress: pbar.update(nbytes)
                f.write(chunk)
        except requests.exceptions.ConnectionError as e:
            fname = url.split('/')[-1]
            timeout_txt =(f'\n Download of {url} has failed after {retries} retries\n'
                          f' Fix the download manually:\n'
                          f'$ mkdir -p \n'
                          f'$ cd \n'
                          f'$ wget -c {url}\n'
                          f'$ tar -zxvf {fname}\n\n'
                          f'And re-run your code once the download is successful\n')
            print(timeout_txt)
            import sys;sys.exit(1)

from typing import *

def listify(o):
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, str): return [o]
    if isinstance(o, Iterable): return list(o)
    return [o]

import re

_camel_re1 = re.compile('(.)([A-Z][a-z]+)')
_camel_re2 = re.compile('([a-z0-9])([A-Z])')
def camel2snake(name):
    s1 = re.sub(_camel_re1, r'\1_\2', name)
    return re.sub(_camel_re2, r'\1_\2', s1).lower()