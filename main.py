import pickle

with open('model/classifier.pkl', 'rb') as f:
    model = pickle.load(f)
