from math import ceil


def get_languages(data):
    languages = data['language'].unique().tolist()
    return languages


def get_language_splitted_data(data):
    languages = get_languages(data)
    for language in languages:
        if language == 'spanish':
            data_spanish = data[data['language'] == 'spanish']
        if language == 'portuguese':
            data_portuguese = data[data['language'] == 'portuguese']
    return data_spanish, data_portuguese


def yield_in_slices(data, n_slices):
    len_data = len(data)

    chunk_size = int(len_data / n_slices)
    data_to_yield = data.copy()
    for _ in range(chunk_size, len_data + 1, chunk_size):
        to_yield = data_to_yield[:chunk_size]
        yield to_yield
        data_to_yield = data_to_yield[chunk_size:]
