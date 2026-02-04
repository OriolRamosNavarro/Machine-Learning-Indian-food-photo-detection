import os
import pickle

with open(os.path.join(os.path.dirname(__file__), '.github/files/features_sift.pickle'), 'rb') as f:
    features_dict = pickle.load(f)

i = 0
for value in features_dict.values():
    i += len(value)

print(i)