import hashlib
from io import BytesIO

import pandas as pd
from pandas import DataFrame

BLOCKSIZE = 65536


def get_dataset_hash(dataset: DataFrame):
    buffer = BytesIO()
    print(dataset)
    dataset.to_parquet(buffer)
    data = buffer.getvalue()

    return get_bytes_hash(data)


def get_bytes_hash(data: bytes):
    hasher = hashlib.sha1()
    for i in range(0, len(data), BLOCKSIZE):
        block = data[i:i + BLOCKSIZE]
        print(len(block))
        hasher.update(block)
    return hasher.hexdigest()


def get_file_hash(path):
    hasher = hashlib.sha1()
    with open(path, 'rb') as afile:
        buf = afile.read(BLOCKSIZE)
        while len(buf) > 0:
            hasher.update(buf)
            buf = afile.read(BLOCKSIZE)
    return hasher.hexdigest()


if __name__ == "__main__":
    data = pd.read_csv('../data/iris.csv')
    hash = get_dataset_hash(data)
    print(hash)
