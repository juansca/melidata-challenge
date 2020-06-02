import pandas as pd
import re
from tqdm import tqdm
from pandarallel import pandarallel
from spacy_utils import get_language_splitted_data, yield_in_slices

import es_core_news_sm as es_spacy
import pt_core_news_sm as pt_spacy
#! python -m spacy download es_core_news_sm
#! python -m spacy download pt_core_news_sm

# cambiar dónde se agrega el __label__ de la categoría
# hacer un transform() para las predicciones

pandarallel.initialize(use_memory_fs=False, progress_bar=False, nb_workers=15)
tqdm.pandas()


N_SLICES = 5

def log_info(f):
    def inner(*args, **kwargs):
        print('Starting to run {}'.format(f.__name__))
        ret = f(*args, **kwargs)
        print('{} finished successfully'.format(f.__name__))

        return ret

    return inner


class SpacyModel():
    def __init__(self, prod_features_path, prod_target_path=None):
        self.prod_f_path = prod_features_path
        self.prod_t_path = prod_target_path
        self.accepted_pos = ['NOUN', 'PROPN', 'VERB', 'ADJ']
        self.forbidden_words = ['oferta', 'vendo', 'permuto', 'envío', 'envio', 'consulte', 'oportunidad', 'se vende', 'gratis', 'local', 'zona']

        self.regex_to_tokenize = self._get_regex()

        self.spacy_models = {'spanish': es_spacy, 'portuguese': pt_spacy}

    def _forbidden_regex(self):
        forbidden = self.forbidden_words
        words = '|'.join(forbidden)
        words_compiled = re.compile(words)

        return words_compiled

    def choose_accepted_words(self, doc):
        pos = self.accepted_pos
        accepted_words = [w.text for w in doc if w.pos_ in pos]
        return " ".join(accepted_words)

    def _get_regex(self):
        # dim = '(x?)(\s?)(\d+)([.|,]?)((\d+)?)(\s?)'
        # dim_2 = dim + '(\s?)(x)(\s?)' + dim
        # dim_3 = dim_2 + '(\s?)(x)(\s?)' + dim
        # ev_dim = '|'.join([dim, dim_2, dim_3])

        # definimos las magnitudes
        magnitud = '(\d+)[.|,]?(\d*)\s*'

        # definimos la unidad
        longitud = 'mm|mtrs|m|cm|km'
        volumen = 'ltrs|l|m3|cc'
        masa = 'kg|grs|gr|g'
        bytes_ = 'b|mb|tb'

        # unidad = '|'.join([longitud, volumen, masa, bytes_])

        # posible union "10cmX15cmx1m"
        union = 'x'

        non_alpha = re.compile(r'[^0-9a-zA-Záéíóú]+|\s+|\n+')
        
        all_metrics = '|'.join([longitud, volumen, masa, bytes_])
        nada_metric = re.compile('(^a-zA-Z.,)(({})+(?!{}){}?)+(?!a-zA-Z)'.format(magnitud, all_metrics, union))
 
        years = re.compile(r'''[0-9]{4}(?!a-zA-Z.,)''')
    
        agregates = [(' ', non_alpha), (' año ', years), (' unknown_metric ', nada_metric)]
        # una forma mas linda
        # # unidades de metricas
        metric_tmpl = '(({})+({})[^0-9]+{}?)+'

        #
        unidades = [(bytes_, ' BYTES '), (volumen, ' VOLUMEN '), (masa, ' MASS '), (longitud, ' LONGITUD ')]
        metricas = []
        for unidad, tag in unidades:
            metrica = metric_tmpl.format(magnitud, unidad, union)
            metricas.append((tag.lower(), re.compile(metrica)))

        return agregates + metricas
    
    
    def tokenize(self, title):
        metrics = self.regex_to_tokenize
        for token, regex in metrics:
            title = re.sub(regex, token, title)
        return title

    @log_info
    def tokenize_title(self, titles_data):
        words_compiled = self._forbidden_regex()

        tokenized_titles = titles_data.progress_apply(lambda x: re.sub(words_compiled, '', x.lower()))
        tokenized_titles = tokenized_titles.progress_apply(lambda x: self.tokenize(x))
        return tokenized_titles

    def edit_category(self, input_data):
        data = input_data
        data['category'] = data['category'].apply(lambda x: '__label__' + x)
        data = data.drop(columns=['label_quality', 'language'])
        return data

    def prepare_data(self, input_data, language, parallel=True):
        data = input_data

        # spanish or portuguese spacy model
        selected_spacy_model = self.spacy_models[language]
        nlp = selected_spacy_model.load()

        if parallel:
            data['spacy_doc'] = data['title'].parallel_apply(lambda x: nlp(x))
        else:
            data['spacy_doc'] = data['title'].progress_apply(lambda x: nlp(x))

        data['title_nouns'] = data['spacy_doc'].apply(self.choose_accepted_words)
        return data

    def save_fasttext_input_file(self, input_data, output_path):
        data = pd.DataFrame()
        data['category'] = input_data['category']
        data['text'] = input_data['title_nouns']
        with open(output_path, 'a') as f:
            data.to_csv(f, sep=' ', header=False, index=False)
        print('File saved successfully.')

    def save_fasttext_predict_file(self, input_data, output_path):
        with open(output_path, 'a') as f:
            input_data.to_csv(f, sep=' ', index=False)
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

    def get_language_split_only_features(self):
        in_f = self.prod_f_path
        products_features = pd.read_csv(in_f)

        spanish, portuguese = get_language_splitted_data(products_features)

        split = {'spanish': spanish, 'portuguese': portuguese}
        return split

    def _batch_train_transform(self, selected_data, language, output_path):
        print('Starting to batch Processing...')
        for batch in yield_in_slices(selected_data, N_SLICES):
            print('Starting new batch')
            prepared_batch = self.edit_category(batch)
            prepared_batch = self.prepare_data(prepared_batch, language)
            self.save_fasttext_input_file(prepared_batch, output_path)
            del(batch)
        print('All batches processed successfully')

    def _batch_predict_transform(self, selected_data, language, output_path):
        print('Starting to batch Processing...')
        for batch in yield_in_slices(selected_data, N_SLICES):
            print('Starting new batch')
            prepared_batch = self.prepare_data(batch, language)
            self.save_fasttext_predict_file(prepared_batch, output_path)
            del(batch)
        print('All batches processed successfully')

    def transform_train(self, language, output_path, batch_proc=True):
        language_split = self.get_language_split()
        selected_data = language_split[language]

        #selected_data['title'] = self.tokenize_title(selected_data.title)

        if batch_proc:
            # Acá vamos a procesar en batches manejables la data
            self._batch_train_transform(selected_data, language, output_path)
        else:
            prepared_data = self.edit_category(selected_data)
            prepared_data = self.prepare_data(prepared_data, language, parallel=False)
            prepared_data['title_nouns'] = self.tokenize_title(prepared_data.title_nouns)
            
            self.save_fasttext_input_file(prepared_data, output_path)

    def transform_predict(self, language, output_path, batch_proc=True):
        language_split = self.get_language_split_only_features()
        selected_data = language_split[language]

        selected_data['title'] = self.tokenize_title(selected_data.title)

        if batch_proc:
            # Acá vamos a procesar en batches manejables la data
            self._batch_predict_transform(selected_data, language, output_path)
        else:
            prepared_data = self.prepare_data(selected_data, language, parallel=False)
        #    prepared_data['title_nouns'] = self.tokenize_title(selected_data.title_nouns)
            self.save_fasttext_predict_file(prepared_data, output_path)

        

if __name__ == '__main__':
    output_spanish_path = 'melifasttext_span_input.csv'
    output_portuguese_path = 'melifasttext_port_input.csv'

    input_f = '../melidata-challenge/data/splitted_dataset/train_0_features_.csv'
    input_t = '../melidata-challenge/data/splitted_dataset/train_0_target_.csv'

    data_model = SpacyModel(input_f, input_t)
    data_model.transform_train('spanish', output_spanish_path)
    data_model.transform_train('portuguese', output_portuguese_path)
