import zipfile
import torch

def read_data(filename):
    z = zipfile.ZipFile(filename, 'r')

    lines = []
    with z.open(z.namelist()[0]) as f:
        i = 0
        for line in f:
            if i % 100 == 0:
                line = line.decode('utf-8').lower().replace("'", " ").replace(".", "").replace("?", "")\
                    .replace("!", "").replace(":", "").replace(";", "")
                lines.append(line)
            i += 1

    z.close()
    return lines