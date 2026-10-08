import numpy as np
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import f1_score
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, f1_score, normalized_mutual_info_score as nmi_score, adjusted_rand_score as ari_score

# def clustering_evaluation(y_true, y_pred):
#     """计算 ACC, NMI, ARI, Macro-F1"""
#     y_true = np.array(y_true)
#     y_pred = np.array(y_pred)
    
#     # 1. 计算 NMI 和 ARI (不需要标签对齐)
#     nmi = NMI(y_true, y_pred)
#     ari = ARI(y_true, y_pred)
    
#     # 2. 匈牙利算法进行标签对齐 (为了计算 ACC 和 F1)
#     D = max(y_pred.max(), y_true.max()) + 1
#     w = np.zeros((D, D), dtype=np.int64)
#     for i in range(y_pred.size):
#         w[y_pred[i], y_true[i]] += 1
    
#     ind_pred, ind_true = linear_sum_assignment(w.max() - w)
    
#     # 构建映射字典
#     map_dict = {ind_pred[i]: ind_true[i] for i in range(len(ind_pred))}
#     y_pred_aligned = np.array([map_dict[p] for p in y_pred])
    
#     # 3. 计算 ACC 和 F1
#     acc = np.sum(y_pred_aligned == y_true) * 1.0 / y_true.size
#     f1 = f1_score(y_true, y_pred_aligned, average='macro')
    
#     return acc, nmi, ari, f1

######################################## Evaluation ########################################
def best_map(y_true, y_pred):
    """
    https://github.com/jundongl/scikit-feature/blob/master/skfeature/utility/unsupervised_evaluation.py
    Permute labels of y_pred to match y_true as much as possible
    """
    if len(y_true) != len(y_pred):
        print("y_true.shape must == y_pred.shape")
        exit(0)

    label_set = np.unique(y_true)
    num_class = len(label_set)

    G = np.zeros((num_class, num_class))
    for i in range(0, num_class):
        for j in range(0, num_class):
            s = y_true == label_set[i]
            t = y_pred == label_set[j]
            G[i, j] = np.count_nonzero(s & t)

    A = linear_sum_assignment(-G)
    new_y_pred = np.zeros(y_pred.shape)
    for i in range(0, num_class):
        new_y_pred[y_pred == label_set[A[1][i]]] = label_set[A[0][i]]
    return new_y_pred.astype(int), label_set[A[1]], label_set[A[0]]

def clustering_evaluation(y_true, y_pred):
    y_pred_, label_original, label_truth = best_map(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred_)
    f1_macro = f1_score(y_true, y_pred_, average='macro')
    # f1_micro = f1_score(y_true, best_map(y_true, y_pred), average='micro')
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    # print('origi label', label_original)
    # print('truth label', label_truth)
    # print('recall', recall_score(y_true, y_pred_, average=None))
    # print('precision', precision_score(y_true, y_pred_, average=None))
    return acc, nmi, ari, f1_macro
