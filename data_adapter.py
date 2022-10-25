
def unpack_data(data):
    # If the input data is not a tuple, return the input data.
    if not isinstance(data, tuple):
        return data, None, None
    elif len(data) == 1:
        return data[0], None, None
    elif len(data) == 2:
        return data[0], data[1], None
    elif len(data) == 3:
        return data[0], data[1], data[2]
    else:
        raise ValueError("Data is expected to be in format `x`, `(x,)`, `(x, y)`, "
                         "or `(x, y, sample_weight)`, found: {}".format(data))
