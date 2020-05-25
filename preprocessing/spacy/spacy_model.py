import pandas as pd
import re
from tqdm import tqdm
from pandarallel import pandarallel
from spacy_utils import get_language_splitted_data, yield_in_slices

import es_core_news_sm as es_spacy
import pt_core_news_sm as pt_spacy
#! python -m spacy download es_core_news_sm
#! python -m spacy download pt_core_news_sm


pandarallel.initialize(use_memory_fs=False, progress_bar=False, nb_workers=15)
tqdm.pandas()


N_SLICES = 5

def log_info(f):
    def inner(*args):
        print('Starting to run {}'.format(f.__name__))
        ret = f(*args)
        print('{} finished successfully'.format(f.__name__))

        return ret

    return inner


class SpacyModel():
    def __init__(self, prod_features_path, prod_target_path):
        self.prod_f_path = prod_features_path
        self.prod_t_path = prod_target_path
        self.accepted_pos = ['NOUN', 'PROPN']

        self.regex_to_tokenize = self._get_regex()
        self.spacy_models = {'spanish': es_spacy, 'portuguese': pt_spacy}

    def choose_nouns(self, doc):
        pos = self.accepted_pos
        nouns = [w.text for w in doc if w.pos_ in pos]
        return " ".join(nouns)

    def _get_regex(self):
        dim = '(x?)(\s?)(\d+)([.|,]?)((\d+)?)(\s?)(\w*)'
        dim_2 = dim + '(\s?)(x)(\s?)' + dim
        dim_3 = dim_2 + '(\s?)(x)(\s?)' + dim
        ev_dim = '|'.join([dim, dim_2, dim_3])

        ev_compiled = re.compile(ev_dim)

        return ev_compiled

    @log_info
    def tokenize_title(self, titles_data):
        ev_compiled = self.regex_to_tokenize

        tokenized_titles = titles_data.progress_apply(lambda x: re.sub(ev_compiled, ' ', x.lower()))
        return tokenized_titles

    @log_info
    def prepare_data(self, input_data, language, parallel=True):
        data = input_data
        data['category'] = data['category'].apply(lambda x: '__label__' + x)
        data = data.drop(columns=['label_quality', 'language'])

        # spanish or portuguese spacy model
        selected_spacy_model = self.spacy_models[language]
        nlp = selected_spacy_model.load()

        if parallel:
            data['spacy_doc'] = data['title'].parallel_apply(lambda x: nlp(x))
        else:
            data['spacy_doc'] = data['title'].progress_apply(lambda x: nlp(x))

        data['title_nouns'] = data['spacy_doc'].apply(self.choose_nouns)
        return data

    def save_fasttext_input_file(self, input_data, output_path):
        data = pd.DataFrame()
        data['category'] = input_data['category']
        data['text'] = input_data['title_nouns']
        with open(output_path, 'a') as f:
            data.to_csv(f, sep=' ', header=False, index=False)
        print('File saved successfully.')

    def _merge(self, features, target):
        products = features.merge(target, on='id', how='inner')
        return products

    @log_info
    def get_language_split(self):
        in_f = self.prod_f_path
        in_t = self.prod_t_path
        products_features = pd.read_csv(in_f)
        products_target = pd.read_csv(in_t)

        products = self._merge(products_features, products_target)
        spanish, portuguese = get_language_splitted_data(products)

        split = {'spanish': spanish, 'portuguese': portuguese}
        return split

    def _batch_transform(self, selected_data, language, output_path):
        print('Starting to batch Processing...')
        for batch in yield_in_slices(selected_data, N_SLICES):
            print('Starting new batch')
            prepared_batch = self.prepare_data(batch, language)
            self.save_fasttext_input_file(prepared_batch, output_path)

        print('All batches processed successfully')

    def transform(self, language, output_path, batch_proc=True):
        language_split = self.get_language_split()
        selected_data = language_split[language]

        selected_data['title'] = self.tokenize_title(selected_data.title)

        if batch_proc:
            # Acá vamos a procesar en batches manejables la data
            self._batch_transform(selected_data, language, output_path)
        else:
            prepared_data = self.prepare_data(selected_data, language, parallel=False)

            self.save_fasttext_input_file(prepared_data, output_path)


if __name__ == '__main__':
    output_spanish_path = 'melifasttext_span_input.csv'
    output_portuguese_path = 'melifasttext_port_input.csv'

    input_f = '../melidata-challenge/data/splitted_dataset/train_0_features_.csv'
    input_t = '../melidata-challenge/data/splitted_dataset/train_0_target_.csv'

    data_model = SpacyModel(input_f, input_t)
    data_model.transform('spanish', output_spanish_path)
    data_model.transform('portuguese', output_portuguese_path)
