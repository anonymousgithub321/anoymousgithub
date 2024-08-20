# The function code for CDRF4QF
import pandas as pd
import numpy as np
import torch
import random
import scipy.sparse as sp
from collections import Iterable, defaultdict
from torch.utils.data import Dataset

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def convert_csr_to_sparse_tensor_inputs(X, device):
    coo = X.tocoo()
    indices = [coo.row, coo.col]
    return torch.LongTensor(indices).to(device), torch.from_numpy(coo.data).to(device), torch.Size(coo.shape)

def csr_to_user_dict(train_matrix):
    """convert a scipy.sparse.csr_matrix to a dict,
    where the key is row number, and value is the
    non-empty index in each row.
    """
    train_dict = {}
    for idx, value in enumerate(train_matrix):
        if len(value.indices):
            train_dict[idx] = value.indices.copy().tolist()
    return train_dict

# 3 ways of pytorch sprase tensor dropout function
# 1. dropout_sparsetensor: only for training
def dropout_sparsetensor(x, keep_prob):
    size = x.size()
    index = x._indices().t()
    values = x._values()
    random_index = torch.rand(len(values)) + keep_prob
    random_index = random_index.int().bool()
    index = index[random_index]
    values = values[random_index]/keep_prob
    g = torch.sparse.FloatTensor(index.t(), values, size)
    return g

# 2. sparse_dropout: only for coo_matrix
def sparse_coo_dropout(x: torch.Tensor, p: float, training: bool):
    x = x.coalesce()
    return torch.sparse_coo_matrix(x.indices(), F.dropout(x.values(), p=p, training=training), size=x.size())

# 3. SparseDropout: only for training
class SparseDropout(torch.nn.Module):
    def __init__(self, kprob=0.5):
        super(SparseDropout, self).__init__()
        self.kprob=kprob

    def forward(self, x):
        mask=((torch.rand(x._values().size())+(self.kprob)).floor()).type(torch.bool)
        rc=x._indices()[:,mask]
        val=x._values()[mask]*(1.0/self.kprob)
        return torch.sparse.FloatTensor(rc, val)

def create_adj_mat(training_user, training_item, n_users, n_items, ssl_ratio, is_subgraph=False, aug_type=1):
    n_nodes = n_users + n_items
    if is_subgraph and ssl_ratio > 0:
        # data augmentation type --- 0: Node Dropout; 1: Edge Dropout; 2: Random Walk
        if aug_type == 0:
            drop_user_idx = randint_choice(n_users, size=n_users * ssl_ratio, replace=False)
            drop_item_idx = randint_choice(n_items, size=n_items * ssl_ratio, replace=False)
            indicator_user = np.ones(n_users, dtype=np.float32)
            indicator_item = np.ones(n_items, dtype=np.float32)
            indicator_user[drop_user_idx] = 0.
            indicator_item[drop_item_idx] = 0.
            diag_indicator_user = sp.diags(indicator_user)
            diag_indicator_item = sp.diags(indicator_item)
            R = sp.csr_matrix((np.ones_like(training_user, dtype=np.float32), (training_user, training_item)), shape=(n_users, n_items))
            R_prime = diag_indicator_user.dot(R).dot(diag_indicator_item)
            (user_np_keep, item_np_keep) = R_prime.nonzero()
            ratings_keep = R_prime.data
            tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep+n_users)), shape=(n_nodes, n_nodes))
        if aug_type in [1, 2]:
            keep_idx = randint_choice(len(training_user), size=int(len(training_user) * (1 - ssl_ratio)), replace=False)
            user_np = np.array(training_user)[keep_idx]
            item_np = np.array(training_item)[keep_idx]
            ratings = np.ones_like(user_np, dtype=np.float32)
            tmp_adj = sp.csr_matrix((ratings, (user_np, item_np+n_users)), shape=(n_nodes, n_nodes))
    else:
        user_np = np.array(training_user)
        item_np = np.array(training_item)
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np+n_users)), shape=(n_nodes, n_nodes))
    adj_mat = tmp_adj + tmp_adj.T

    # pre adjcency matrix
    rowsum = np.array(adj_mat.sum(1))
    d_inv = np.power(rowsum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    norm_adj_tmp = d_mat_inv.dot(adj_mat)
    adj_matrix = norm_adj_tmp.dot(d_mat_inv)
    # print('use the pre adjcency matrix')

    return adj_matrix

def pairwise_sampling_v2(user_pos_dict, num_samples, num_item):
    user_arr = np.array(list(user_pos_dict.keys()), dtype=np.int32)
    user_idx = randint_choice(len(user_arr), size=num_samples, replace=True)
    users_list = user_arr[user_idx]

    # count the number of each user, i.e., the numbers of positive and negative items for each user
    user_pos_len = defaultdict(int)
    for u in users_list:
        user_pos_len[u] += 1

    user_pos_sample = dict()
    user_neg_sample = dict()
    for user, pos_len in user_pos_len.items():
        pos_items = user_pos_dict[user]
        pos_idx = randint_choice(len(pos_items), size=pos_len, replace=True)
        pos_idx = pos_idx if isinstance(pos_idx, Iterable) else [pos_idx]
        user_pos_sample[user] = list(pos_items[pos_idx])

        neg_items = randint_choice(num_item, size=pos_len, replace=True, exclusion=user_pos_dict[user])
        user_neg_sample[user] = neg_items if isinstance(neg_items, Iterable) else [neg_items]

    pos_items_list = [user_pos_sample[user].pop() for user in users_list]
    neg_items_list = [user_neg_sample[user].pop() for user in users_list]

    return users_list, pos_items_list, neg_items_list

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        return [col[idx] for col in self.data]

def pad_sequences(sequences, value=0., max_len=None,
                  padding='post', truncating='post', dtype=np.int32):
    """Pads sequences to the same length.

    Args:
        sequences (list): A list of lists, where each element is a sequence.
        value (int or float): Padding value. Defaults to `0.`.
        max_len (int or None): Maximum length of all sequences.
        padding (str): `"pre"` or `"post"`: pad either before or after each
            sequence. Defaults to `post`.
        truncating (str): `"pre"` or `"post"`: remove values from sequences
            larger than `max_len`, either at the beginning or at the end of
            the sequences. Defaults to `post`.
        dtype (int or float): Type of the output sequences. Defaults to `np.int32`.

    Returns:
        np.ndarray: Numpy array with shape `(len(sequences), max_len)`.

    Raises:
        ValueError: If `padding` or `truncating` is not understood.
    """
    if max_len is None:
        max_len = np.max([len(x) for x in sequences])

    x = np.full([len(sequences), max_len], value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-max_len:]
        elif truncating == 'post':
            trunc = s[:max_len]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x

def pad_sequences_3d(sequences, value=0., max_len=None,
                    padding='post', truncating='post', dtype=np.int32):
    if max_len is None:
        max_len = np.max([len(x) for x in sequences])

    x = np.full([len(sequences), max_len, len(sequences[0][0])], value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-max_len:]
        elif truncating == 'post':
            trunc = s[:max_len]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        if padding == 'post':
            x[idx, :len(trunc), :] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):, :] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x






