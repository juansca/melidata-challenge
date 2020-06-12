# quiero tomar un archivo con categorías y títulos,
# 'unir' todos los títulos, hacer el word counts de eso
# y crear el archivo de salida con formato jsonlines

import pandas as pd
import json
from tqdm import tqdm
from pandarallel import pandarallel
from collections import Counter

tqdm.pandas()
pandarallel.initialize(use_memory_fs=False, progress_bar=False, nb_workers=15)

def word_counter(input_f, output_path):
    data = pd.read_csv(input_f, names=['category', 'title'], sep=' ', header=None)
    data['category'] = data['category'].progress_apply(lambda x: x.split('__label__')[1])

    grouped_data = data.groupby('category').agg(' '.join)
    grouped_data['word_count'] = grouped_data['title'].parallel_apply(lambda x: Counter(x.split()))

    with open(output_path, 'w') as f:
        for category, word_count in zip(grouped_data.index, grouped_data['word_count']):
            line_dict = {'category': category, 'word_count': word_count}
            f.write(json.dumps(line_dict) + '\n')
