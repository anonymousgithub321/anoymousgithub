# The evaluation code for CDRF4QF
import numpy as np
from collections import defaultdict
import numba as nb
from time import time

@nb.njit('int32[:,::1](float64[:,::1])', parallel=True)
def fastSort(a):
    b = np.empty(a.shape, dtype=np.int32)
    for i in nb.prange(a.shape[0]):
        b[i,:] = np.argsort(a[i,:])
    return b

def evaluate11(posprobe, r, k):  # 采用numba加速，加入MRR，NDCG, HR,传入的评分矩阵已经去掉了训练集的正样本
    userlist = list(posprobe.keys())
    # for user in userlist:
    #     r[user, postrain[user]] = -9999

    r[userlist, :] = fastSort(r[userlist, :])
    pred = r[:, ::-1][:, 0:k[-1]]

    recall = []
    precision = []
    map = []
    # mrr = []
    ndcg = []
    # hr = []
    for kk in k:
        recall_tmp = []
        precision_tmp = []
        map_tmp = []
        # mrr_tmp = []
        ndcg_tmp = []
        # hr_tmp = []
        rank = np.arange(1, kk + 1)
        scores = np.log2(np.arange(2, kk + 2))
        for user in userlist:
            predict_tmp = np.zeros(kk, dtype=np.float)
            boo_tmp = np.zeros(kk, dtype=np.float)
            num_pos = len(posprobe[user])
            max_r = np.minimum(kk, num_pos)
            max_r_vector = np.zeros(kk, dtype=np.float)
            max_r_vector[:max_r] = 1
            ll = 1
            for l in range(kk):
                if pred[user, l] in posprobe[user]:
                    predict_tmp[l] = ll
                    boo_tmp[l] = 1
                    ll += 1
            sum_tmp = np.sum(boo_tmp)
            recall_tmp.append(sum_tmp / num_pos)
            # recall_tmp.append(sum_tmp / max_r)
            precision_tmp.append(sum_tmp / kk)
            map_tmp.append(np.sum(predict_tmp / rank) / kk)
            # mrr_tmp.append(np.sum(boo_tmp/rank))
            # hr_tmp.append(np.float(sum_tmp > 0))
            idcg = np.sum(max_r_vector/scores)
            dcg = np.sum(boo_tmp/scores)
            # idcg[idcg == 0.] = 1.
            ndcg_tmp.append(dcg/idcg)

        recall.append(np.mean(recall_tmp))
        precision.append(np.mean(precision_tmp))
        map.append(np.mean(map_tmp))
        # mrr.append(np.mean(mrr_tmp))
        # hr.append(np.mean(hr_tmp))
        ndcg.append(np.mean(ndcg_tmp))

    # return recall, precision, map, mrr, ndcg, hr
    return recall, precision, map, ndcg

def evaluate12(posprobe, st_idx, end_idx, r, k):  # 采用分batch运算，输入开始和结束的用户id，同时采用numba加速
    userlist = list(set(posprobe.keys()).intersection(set(range(st_idx, end_idx))))

    r = fastSort(r)
    pred = r[:, ::-1][:, 0:k[-1]]

    recall = []
    precision = []
    map = []
    # mrr = []
    ndcg = []
    # hr = []
    for kk in k:
        recall_tmp = []
        precision_tmp = []
        map_tmp = []
        # mrr_tmp = []
        ndcg_tmp = []
        # hr_tmp = []
        rank = np.arange(1, kk + 1)
        scores = np.log2(np.arange(2, kk + 2))
        for user in userlist:
            predict_tmp = np.zeros(kk, dtype=np.float)
            boo_tmp = np.zeros(kk, dtype=np.float)
            num_pos = len(posprobe[user])
            max_r = np.minimum(kk, num_pos)
            max_r_vector = np.zeros(kk, dtype=np.float)
            max_r_vector[:max_r] = 1
            ll = 1
            for l in range(kk):
                if pred[user-st_idx, l] in posprobe[user]:
                    predict_tmp[l] = ll
                    boo_tmp[l] = 1
                    ll += 1
            sum_tmp = np.sum(boo_tmp)
            recall_tmp.append(sum_tmp / num_pos)
            # recall_tmp.append(sum_tmp / max_r)
            precision_tmp.append(sum_tmp / kk)
            map_tmp.append(np.sum(predict_tmp / rank) / kk)
            # mrr_tmp.append(np.sum(boo_tmp/rank))
            # hr_tmp.append(np.float(sum_tmp > 0))
            idcg = np.sum(max_r_vector/scores)
            dcg = np.sum(boo_tmp/scores)
            # idcg[idcg == 0.] = 1.
            ndcg_tmp.append(dcg/idcg)

        recall.append(np.sum(recall_tmp))
        precision.append(np.sum(precision_tmp))
        map.append(np.sum(map_tmp))
        # mrr.append(np.sum(mrr_tmp))
        # hr.append(np.sum(hr_tmp))
        ndcg.append(np.sum(ndcg_tmp))

    # return recall, precision, map, ndcg
    return np.array(recall), np.array(precision), np.array(map), np.array(ndcg)

