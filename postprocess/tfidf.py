import pandas as pd
import json
from tqdm import tqdm
#from pandarallel import pandarallel
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

tqdm.pandas()
#pandarallel.initialize(use_memory_fs=False, progress_bar=False, nb_workers=15)
tfidf_vectorizer = TfidfVectorizer(use_idf=True)

def get_tifidf(input_f, tfidf_output_path, idf_output_path):
    data = pd.read_csv(input_f, names=['category', 'title'], sep=' ', header=None)
    # data['category'] = data['category'].progress_apply(lambda x: x.split('__label__')[1])

    # junto todos los títulos en una sola línea, separados por un espacio
    grouped_data = data.groupby('category').agg(' '.join)

    # armo una lista con los títulos, esto es lo que come el tfidf vectorizer
    titles_list = grouped_data['title'].tolist()
    # genero la matriz con los tfidf
    tfidf_vectors = tfidf_vectorizer.fit_transform(titles_list)
    # convierto esa matriz en una lista de listas (c/ fila de la matriz es un elemento)
    tfidf_vectors_list = tfidf_vectors.T.todense().transpose().tolist()
    # agrego esto al df con la data agrupada
    grouped_data['tfidf_vector'] = tfidf_vectors_list
    # creo una lista con las palabras
    words = tfidf_vectorizer.get_feature_names()
    # creo una nueva columna en el df con un diccionario que tiene las palabras con
    # su tfidf correspondiente
    grouped_data['tfidf_dict'] = grouped_data['tfidf_vector'].apply(lambda x: dict(zip(words, x)))

    with open(tfidf_output_path, 'w') as f:
        for category, tfidf_dict in zip(grouped_data.index, grouped_data['tfidf_dict']):
            line_dict = {'category': category, 'tfidf': tfidf_dict}
            f.write(json.dumps(line_dict) + '\n')

    with open(idf_output_path, 'w') as f:
        idf_dict = dict(zip(words, tfidf_vectorizer.idf_.tolist()))
        f.write(json.dump(idf_dict))

# después para ver la distancia me debería hacer una función que calcule la distancia coseno
# i.e. cos(theta) = dot(A,B)/(norm(A)*norm(B))

def get_cosine_distance(dic1, dic2):
    num = 0
    den1 = 0
    for key1, val1 in dic1:
        num += val1*dic2.get(key1,0.0)
        den1 += val1*val1
    den2 = 0
    for val2 in dic2.values():
        den2 += val2*val2
    return num/math.sqrt(den1*den2)

def get_single_tfidf(self, title, idf_path):
    with open(idf_path, 'r') as f1:
        idfs = json.load(f1)
    word_counts = Counter(title.split())
    tfidf = {}
    for key1, val1 in words_counts:
        tfidf[key1] = val1*idfs.get(key1,0.0)
    return tfidf
    # aca tengo que agarrar el titulo, hacerle el wordcounts y
    # multiplicar cada uno por el idf correspondiente.
    # si la palabra no está, multiplicar por cero

def get_closest_cat(self, title, language, k, tfidf_path, idf_path):
    try:
        tfidf = get_single_tfidf(title, idf_path)
        predictions = self.models[language].predict(title, k=k)
        with open(tfidf_path, 'r') as f2:
            cat_tfidf = pd.read_json(tfidf_path, line=True)
        cat_tfidf['category'] = cat_tfidf['category'].progress_apply(lambda x: x.split('__label__')[1])
        categories = predictions[:][0]
        distances = {}
        for category in categories:
            category_tfidf = cat_tfidf[cat_tfidf['category'] == category]['tfidf']
            distances[category] = get_cosine_distance(tfidf, category_tfidf)
        return min(distances, key=distances.get)

    except KeyError:
        msg = 'Model for selected language ({}) is not available'
        raise KeyError(msg.format(language))
