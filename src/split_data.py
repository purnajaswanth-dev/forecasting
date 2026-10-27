def split_data(data):

    data = data.sort_values("Timestamp")

    n = len(data)

    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    train = data.iloc[:train_end]

    val = data.iloc[train_end:val_end]

    test = data.iloc[val_end:]

    return train, val, test