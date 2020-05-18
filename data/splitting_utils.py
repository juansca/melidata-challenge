from sklearn.model_selection import StratifiedShuffleSplit

def get_feature_target_spliting(dataset):
    # Create ids to compute metric later
    ids = pd.DataFrame(train_data.index, columns=['id'])
    
    dataset['id'] = ids
    X = dataset[['id', 'title', 'label_quality', 'language']]
    y = dataset[['id', 'category']]
    return X, y

def train_validation_split(dataset, n_splits=1, test_size=None):
    # Split in to X, y dataframes
    X, y = get_feature_target_spliting(dataset)
    
    # Strattified train-test splitting
    stratified_split = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=0)
    
    # Hacemos esto para que el stratified se haga sobre las categorias
    y_cat = y.category
    for train_index, test_index in stratified_split.split(X, y_cat):
        
        X_train, X_test = X.loc[train_index, :], X.loc[test_index, :]
        y_train, y_test = y.loc[train_index, :], y.loc[test_index, :]
        
        yield X_train, X_test, y_train, y_test