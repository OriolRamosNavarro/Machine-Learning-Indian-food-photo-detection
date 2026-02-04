from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, MiniBatchKMeans
import pickle
import os
import numpy as np
import json

def convertir_a_arrays(data_dict):
    for clase in data_dict:
        for img in data_dict[clase]:
            if not isinstance(data_dict[clase][img], np.ndarray):
                data_dict[clase][img] = np.array(data_dict[clase][img])
    return data_dict

def split_train_test(flag, method, dim_descriptors, max_keypoints, test_size=0.2, random_state=42):
    if flag == False:
        print("Split ja fet, carregant desde json...")
        with open(os.path.join(os.path.dirname(__file__), f'train_data_{method}.pickle'), 'rb') as f:
            train_dict = pickle.load(f)
        with open(os.path.join(os.path.dirname(__file__), f'test_data_{method}.pickle'), 'rb') as f:
            test_dict = pickle.load(f)
    else:       
        with open(os.path.join(os.path.dirname(__file__),
                               f'features{method}_dim{dim_descriptors}_maxkeypoints{max_keypoints}.pickle'), 'rb') as f:
            features_dict = pickle.load(f)

        train_dict = {}
        test_dict = {}
        
        for food_name, images_dict in features_dict.items():
            paths = list(images_dict.keys())
            
            train_paths, test_paths = train_test_split(
                paths, 
                test_size=test_size, 
                random_state=random_state
            )
            
            train_dict[food_name] = {p: images_dict[p] for p in train_paths}
            test_dict[food_name] = {p: images_dict[p] for p in test_paths}
            
        with open(os.path.join(os.path.dirname(__file__), f'train_data_{method}.pickle'), 'wb') as f:
            pickle.dump(train_dict, f)
        with open(os.path.join(os.path.dirname(__file__), f'test_data_{method}.pickle'), 'wb') as f:
            pickle.dump(test_dict, f)

    for key in train_dict.keys():
        print(f"{key}: {len(train_dict[key])} train, {len(test_dict[key])} test")
    
    return train_dict, test_dict


def build_vocabulary(train_dict, method, flag, K):
    valores = list(train_dict.values())
    if isinstance(valores[0], np.ndarray):
        pass
    else:
        train_dict = convertir_a_arrays(train_dict)


    if flag == False:
        print("Carregant el kmeans des del pickle")
        with open(os.path.join(os.path.dirname(__file__), f'kmeans_{method}_{K}.pickle'), 'rb') as f:
            kmeans = pickle.load(f)
    else:
        all_descriptors = []
        
        for food_name, images_dict in train_dict.items():
            for img_path, descriptors in images_dict.items():
                all_descriptors.append(descriptors)
        
        all_descriptors_numpy = np.vstack(all_descriptors)
        print(f"Total descriptors: {all_descriptors_numpy.shape}")
        
        print(f"Entrenando K-Means con K={K}...")
        kmeans = MiniBatchKMeans(
        n_clusters=K,
        init='k-means++',
        n_init=5,                  
        batch_size=10000,
        max_iter=100,
        random_state=42,
        verbose = True )
        kmeans.fit(all_descriptors_numpy)
        with open(os.path.join(os.path.dirname(__file__), f'kmeans_{method}_{K}.pickle'), 'wb') as f:
           pickle.dump(kmeans, f)

        
        print(f"Vocabulario creado: {kmeans.cluster_centers_.shape}")
        
    return kmeans

def vocabulary_crear(flag_split, flag_kmeans, k, method = 'sift'):
    train_data, test_data = split_train_test(flag_split, method)
    kmeans = build_vocabulary(train_data, method, flag_kmeans, k)
    return train_data, test_data, kmeans