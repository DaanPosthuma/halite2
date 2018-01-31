import pickle

data_folder = 'data'


def dump(obj, file_name):
    pickle.dump(obj, open(data_folder + '/' + file_name, 'wb'))


def load(file_name):
    return pickle.load(open(data_folder + '/' + file_name, 'rb'))
