import pandas as pd
import fasttext


def log_info(f):
    def inner(*args):
        print('Starting to run {}'.format(f.__name__))
        f(*args)
        print('{} finished successfully'.format(f.__name__))

    return inner



class FasttextModel():

    def __init__(self):
        self.models = {}

    @log_info
    def _train_unilanguage_model(self, data_path, language, **kwargs):
        model = fasttext.train_supervised(data_path, **kwargs)
        self.models[language] = model

    def fit(self, span_data_path=None, port_data_path=None):
        if span_data_path:
            self._train_unilanguage_model(span_data_path, 'spanish')
        if port_data_path:
            self._train_unilanguage_model(port_data_path, 'portuguese')

    def predict(self, title, language, k=1):
        try:
            prediction = self.models[language].predict(title, k=k)
        except KeyError:
            msg = 'Model for selected language ({}) is not available'
            raise KeyError(msg.format(language))

        return prediction

    def save(self, dir_model):
        for language, model in models.items():
            model.save_model(dir_model + language + '.bin')

    def load(self, dir_model):
        pass
