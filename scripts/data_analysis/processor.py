import pandas as pd


def read_csv(path, chunk_size=10000):
    chunks = pd.read_csv(path, chunksize=chunk_size, low_memory=False)
    for block in chunks:
        yield block


def filtered_data(chunks):
    for data in chunks:
        dept = None
        date = None

        for name in data.columns:
            if 'Department' in name:
                dept = name
            if 'Object Begin Date' in name:
                date = name
        if dept is None or date is None:
            continue

    yield data[[dept, date]].copy()


def clean_data(chunks):
    for data in chunks:
        data.columns = ['Department', 'Date']
        data['Date'] = pd.to_numeric(data['Date'], errors='coerce')
        data = data.dropna()
        data['Date'] = data['Date'].astype(int)

        yield data
