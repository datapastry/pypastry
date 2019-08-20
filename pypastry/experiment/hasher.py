import hashlib
from io import BytesIO

import pandas as pd
from pandas import DataFrame

BLOCKSIZE = 65536


def get_dataset_hash(dataset: DataFrame):
    buffer = BytesIO()
    dataset.to_parquet(buffer)
    data = buffer.getvalue()

    return get_bytes_hash(data)


def get_bytes_hash(data: bytes):
    hasher = hashlib.sha1()
    for i in range(0, len(data), BLOCKSIZE):
        block = data[i:i + BLOCKSIZE]
        hasher.update(block)
    return hasher.hexdigest()


if __name__ == "__main__":
    data = pd.read_csv('../data/iris.csv')
    hash = get_dataset_hash(data)
    print(hash)
