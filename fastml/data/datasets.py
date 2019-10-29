from fastml.core import *

DEFAULT_LOCATION = os.path.expanduser(os.getenv('FASTML_HOME', '~/.fastml'))

def url2name(url): return url.split('/')[-1]

def datapath4file(filename):
    "Return data path to `filename`, checking locally first then in the config file."
    local_path = Path(DEFAULT_LOCATION)/'data'/filename
    return local_path

def default_data_path(url, ext:str='.tgz'):
    return datapath4file(f'{url2name(url)}{ext}')

def download_data(url:str, fname:str=None, ext:str='.tgz') -> Path:
    "Download `url` to destination `fname`."
    fname = Path(ifnone(fname, default_data_path(url, ext=ext)))
    os.makedirs(fname.parent, exist_ok=True)
    if not fname.exists():
        print(f'Downloading {url}')
        download_url(f'{url}{ext}', fname)
    return fname

def normalize(x, mean, std): return (x-mean)/std

def normalize_data(x_train, y_train, x_valid, y_valid):
    train_mean,train_std = x_train.mean(),x_train.std()
    x_train = normalize(x_train, train_mean, train_std)
    x_valid = normalize(x_valid, train_mean, train_std)
    return x_train, y_train, x_valid, y_valid

def get_data(path):
    with gzip.open(path, 'rb') as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')
    return map(tensor, (x_train,y_train,x_valid,y_valid))

class Dataset():
    def __init__(self, x, y): self.x,self.y = x,y
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i],self.y[i]

class DataLoader():
    def __init__(self, dataset, batch_size): self.dataset, self.batch_size = dataset, batch_size
    def __iter__(self):
        for batch_num in range(0, len(self.dataset), self.batch_size):
            yield self.dataset[batch_num : batch_num + self.batch_size]

def wih_data_loader(train_ds, valid_ds, batch_size):
    return DataLoader(train_ds, batch_size), DataLoader(valid_ds, batch_size)

class Datasets:

    @classmethod
    def MNIST(cls):
        path = download_data('http://deeplearning.net/data/mnist/mnist.pkl', ext='.gz')
        x_train, y_train, x_valid, y_valid = get_data(path)
        x_train, y_train, x_valid, y_valid = normalize_data(x_train, y_train, x_valid, y_valid)
        return Dataset(x_train, y_train), Dataset(x_valid, y_valid)