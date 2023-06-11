import numpy as np
import random
import torch
import pandas as pd
import dgl
import networkx as nx
from sklearn.model_selection import StratifiedKFold

device = torch.device('cuda')

def get_adj(edges, size):
    edges_tensor = torch.LongTensor(edges).t()
    values = torch.ones(len(edges))
    adj = torch.sparse.LongTensor(edges_tensor, values, size).to_dense().long()
    return adj


def k_matrix(matrix, k):
    num = matrix.shape[0]
    knn_graph = np.zeros(matrix.shape)
    idx_sort = np.argsort(-(matrix - np.eye(num)), axis=1)
    for i in range(num):
        knn_graph[i, idx_sort[i, :k + 1]] = matrix[i, idx_sort[i, :k + 1]]
        knn_graph[idx_sort[i, :k + 1], i] = matrix[idx_sort[i, :k + 1], i]
    return knn_graph + np.eye(num)


def get_data(args):
    data = dict()

    drf = pd.read_csv(args.data_dir + 'DrugFingerprint.csv').iloc[:, 1:].to_numpy()
    drg = pd.read_csv(args.data_dir + 'DrugGIP.csv').iloc[:, 1:].to_numpy()

    dip = pd.read_csv(args.data_dir + 'DiseasePS.csv').iloc[:, 1:].to_numpy()
    dig = pd.read_csv(args.data_dir + 'DiseaseGIP.csv').iloc[:, 1:].to_numpy()

    data['drug_number'] = int(drf.shape[0])
    data['disease_number'] = int(dig.shape[0])

    data['drf'] = drf
    data['drg'] = drg
    data['dip'] = dip
    data['dig'] = dig

    data['drdi'] = pd.read_csv(args.data_dir + 'DrugDiseaseAssociationNumber.csv', dtype=int).to_numpy()
    data['drpr'] = pd.read_csv(args.data_dir + 'DrugProteinAssociationNumber.csv', dtype=int).to_numpy()
    data['dipr'] = pd.read_csv(args.data_dir + 'ProteinDiseaseAssociationNumber.csv', dtype=int).to_numpy()

    data['drugfeature'] = pd.read_csv(args.data_dir + 'Drug_mol2vec.csv', header=None).iloc[:, 1:].to_numpy()
    data['diseasefeature'] = pd.read_csv(args.data_dir + 'DiseaseFeature.csv', header=None).iloc[:, 1:].to_numpy()
    data['proteinfeature'] = pd.read_csv(args.data_dir + 'Protein_ESM.csv', header=None).iloc[:, 1:].to_numpy()
    data['protein_number']= data['proteinfeature'].shape[0]

    return data


def data_processing(data, args):
    drdi_matrix = get_adj(data['drdi'], (args.drug_number, args.disease_number))
    one_index = []
    zero_index = []
    for i in range(drdi_matrix.shape[0]):
        for j in range(drdi_matrix.shape[1]):
            if drdi_matrix[i][j] >= 1:
                one_index.append([i, j])
            else:
                zero_index.append([i, j])
    random.seed(args.random_seed)
    random.shuffle(one_index)
    random.shuffle(zero_index)

    unsamples = zero_index[int(args.negative_rate * len(one_index)):]
    data['unsample'] = np.array(unsamples)

    zero_index = zero_index[:int(args.negative_rate * len(one_index))]

    index = np.array(one_index + zero_index, dtype=int)
    label = np.array([1] * len(one_index) + [0] * len(zero_index), dtype=int)
    samples = np.concatenate((index, np.expand_dims(label, axis=1)), axis=1)
    label_p = np.array([1] * len(one_index), dtype=int)

    drdi_p = samples[samples[:, 2] == 1, :]
    drdi_n = samples[samples[:, 2] == 0, :]

    drs_mean = (data['drf'] + data['drg']) / 2
    dis_mean = (data['dip'] + data['dig']) / 2

    drs = np.where(data['drf'] == 0, data['drg'], drs_mean)
    dis = np.where(data['dip'] == 0, data['dip'], dis_mean)

    data['drs'] = drs
    data['dis'] = dis
    data['all_samples'] = samples
    data['all_drdi'] = samples[:, :2]
    data['all_drdi_p'] = drdi_p
    data['all_drdi_n'] = drdi_n
    data['all_label'] = label
    data['all_label_p'] = label_p

    return data


def k_fold(data, args):
    k = args.k_fold
    skf = StratifiedKFold(n_splits=k, random_state=None, shuffle=False)
    X = data['all_drdi']
    Y = data['all_label']
    # n = skf.get_n_splits(X, Y)
    X_train_all, X_test_all, Y_train_all, Y_test_all = [], [], [], []
    for train_index, test_index in skf.split(X, Y):
        # print('Train:', train_index, 'Test:', test_index)
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        Y_train = np.expand_dims(Y_train, axis=1).astype('float64')
        Y_test = np.expand_dims(Y_test, axis=1).astype('float64')
        X_train_all.append(X_train)
        X_test_all.append(X_test)
        Y_train_all.append(Y_train)
        Y_test_all.append(Y_test)

    for i in range(k):
        X_train1 = pd.DataFrame(data=np.concatenate((X_train_all[i], Y_train_all[i]), axis=1), columns=['drug', 'disease', 'label'])
        X_train1.to_csv(args.data_dir + 'fold/' + str(i) + '/data_train.csv')
        X_test1 = pd.DataFrame(data=np.concatenate((X_test_all[i], Y_test_all[i]), axis=1), columns=['drug', 'disease', 'label'])
        X_test1.to_csv(args.data_dir + 'fold/' + str(i) + '/data_test.csv')

    data['X_train'] = X_train_all
    data['X_test'] = X_test_all
    data['Y_train'] = Y_train_all
    data['Y_test'] = Y_test_all
    return data


def dgl_similarity_graph(data, args):
    drdr_matrix = k_matrix(data['drs'], args.neighbor)
    didi_matrix = k_matrix(data['dis'], args.neighbor)
    drdr_nx = nx.from_numpy_matrix(drdr_matrix)
    didi_nx = nx.from_numpy_matrix(didi_matrix)
    drdr_graph = dgl.from_networkx(drdr_nx)
    didi_graph = dgl.from_networkx(didi_nx)

    drdr_graph.ndata['drs'] = torch.tensor(data['drs'])
    didi_graph.ndata['dis'] = torch.tensor(data['dis'])

    return drdr_graph, didi_graph, data


def dgl_heterograph(data, drdi, args):
    drdi_list, drpr_list, dipr_list = [], [], []
    for i in range(drdi.shape[0]):
        drdi_list.append(drdi[i])
    for i in range(data['drpr'].shape[0]):
        drpr_list.append(data['drpr'][i])
    for i in range(data['dipr'].shape[0]):
        dipr_list.append(data['dipr'][i])

    node_dict = {
        'drug': args.drug_number,
        'disease': args.disease_number,
        'protein': args.protein_number
    }

    heterograph_dict = {
        ('drug', 'association', 'disease'): (drdi_list),
        ('drug', 'association', 'protein'): (drpr_list),
        ('disease', 'association', 'protein'): (dipr_list)
    }

    data['feature_dict'] ={
        'drug': torch.tensor(data['drugfeature']),
        'disease': torch.tensor(data['diseasefeature']),
        'protein': torch.tensor(data['proteinfeature'])
    }

    drdipr_graph = dgl.heterograph(heterograph_dict, num_nodes_dict=node_dict)

    return drdipr_graph, data





