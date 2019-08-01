import hashlib

BLOCKSIZE = 65536
hasher = hashlib.sha1()


def get_file_hash(path):
    with open(path, 'rb') as afile:
        buf = afile.read(BLOCKSIZE)
        while len(buf) > 0:
            hasher.update(buf)
            buf = afile.read(BLOCKSIZE)
    return hasher.hexdigest()
