# -*- coding: UTF-8 -*-
"""
Created by louis at 2021/9/13
Description:
"""
import os
import gc
import glob
import torch
from pandas import DataFrame
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import time

from itertools import islice
from torch.utils.data import Dataset, DataLoader

from multiprocessing import Pool
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

from tqdm.auto import tqdm
import logging
import resource
import shutil
import os




rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

datefmt = '%Y-%m-%d %H:%M:%S'
logging.basicConfig(filename='pytorch-baseline.log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt=datefmt, level=logging.DEBUG)

# import tqdm

tqdm.pandas()

import warnings

from multiprocessing import cpu_count


def get_path_dict(f, v):
    f_dict = {}
    for i in tqdm(v):
        fpath = f'{f}/stock_id={i}'
        flist = glob.glob(os.path.join(fpath, '*.parquet'))

        if len(flist) > 0:
            f_dict[i] = flist[0]

    return f_dict


# train_idx, valid_idx = train_test_split(train_ds['row_id'], shuffle=True, test_size=0.1, random_state=SEED)

# ds： train.csv里面的数据 f_dict：是 book_train.parquet 里面的数据
def process_optiver_ds(ds, f_dict, skip_cols, t_dict):
    x = []
    y = []
    full_seconds_in_bucket = {'seconds_in_bucket': np.arange(600)}
    full_seconds_in_bucket = pd.DataFrame(full_seconds_in_bucket)

    for stock_id, stock_fnmame in tqdm(f_dict.items()):
        trade_train_ = t_dict.get(stock_id)
        trade_train_ = pd.read_parquet(trade_train_)
        optiver_ds = pd.read_parquet(stock_fnmame)

        time_ids = optiver_ds['time_id'].unique()
        for time_id in time_ids:
            optiver_ds_ = optiver_ds[optiver_ds['time_id'] == time_id]
            optiver_ds_ = pd.merge(full_seconds_in_bucket, optiver_ds_, how='left', on='seconds_in_bucket')
            optiver_ds_ = pd.merge(optiver_ds_, trade_train_[trade_train_['time_id'] == time_id], how='left',
                                   on='seconds_in_bucket')
            # optiver_ds_.drop(skip_cols)
            optiver_ds_.drop(['time_id_x', 'time_id_y'], axis=1)
            optiver_ds_ = np.nan_to_num(optiver_ds_)
            row_id = str(stock_id) + '-' + time_id.astype(str)
            r = ds[ds['row_id'] == row_id]['target']
            x.append(optiver_ds_)
            y.append(r)

    return x, y


def chunks(data, SIZE=10000):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k: data[k] for k in islice(it, SIZE)}


def process_book_train_chunk(chunk_ds):
    return process_optiver_ds(train_ds, chunk_ds, book_skip_columns, trade_train_dict)


def process_book_test_chunk(chunk_ds):
    return process_optiver_ds(test_ds, chunk_ds, book_skip_columns, trade_test_dict)


'''

# 将样本分成4块，每块里面有28条数据
book_train_chunks = [i for i in chunks(book_train_dict, int(len(book_train_dict) / NTHREADS))]
# trade_train_chunks = [i for i in chunks(trade_train_dict, int(len(trade_train_dict) / NTHREADS))]

z = 1 if len(book_test_dict) < NTHREADS else NTHREADS
book_test_chunks = [i for i in chunks(book_test_dict, int(len(book_test_dict) / z))]
# trade_test_chunks = [i for i in chunks(trade_test_dict, int(len(trade_test_dict) / z))]

pool = Pool(NTHREADS)  # 创建进程池，最大进程数为 NTHREADS
r = pool.map(process_book_train_chunk, book_train_chunks)
pool.close()

a1, a2 = zip(*r)

pool = Pool(NTHREADS)  # 创建进程池，最大进程数为 NTHREADS
r = pool.map(process_book_test_chunk, book_test_chunks)
pool.close()

t_a1, t_a2 = zip(*r)

np_train = a1
np_target = a2'''


# Scaler
# transformers = []
# for i in tqdm(range(np_train.shape[1])):
#     a = np.nan_to_num(np_train[train_idx])
#     b = np.nan_to_num(np_train[valid_idx])
#
#     transformer = StandardScaler()  # StandardScaler is very useful!
#     np_train[train_idx] = transformer.fit_transform(a)
#     np_train[valid_idx] = transformer.transform(b)
#     transformers.append(transformer)  # Save Scalers for the inference stage


class LSTMModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.1):
        super(LSTMModel, self).__init__()
        # self.drop = nn.Dropout(dropout)
        # self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp + input_features_num, nhid + input_features_num, nlayers, dropout=dropout,
                           batch_first=True, bidirectional=True)
        self.regress_rnn = nn.Linear(2 * nhid + 2 * input_features_num, 1)
        self.decoder = nn.Sequential(
            nn.Linear(3 * nhid + 2 * input_features_num, nhid + input_features_num),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(nhid + input_features_num, ntoken),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(ntoken, 1),
        )
        self.self_attention = nn.Sequential(
            nn.Linear(3 * nhid + 2 * input_features_num, 10 * (nhid + input_features_num)),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(10 * (nhid + input_features_num), 10 * (nhid + input_features_num)),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(10 * (nhid + input_features_num), 3 * nhid + 2 * input_features_num),
            nn.Softmax(dim=1)
        )
        # self.decoder_1 = nn.Linear(nhid, ntoken)
        # self.decoder_2 = nn.Linear(ntoken, 1)
        self.conv1d_relu_stack = nn.Sequential(
            nn.Conv1d(in_channels=600, out_channels=1200, kernel_size=3),
            nn.Dropout(0.1),
            nn.ReLU(),  # 9
            nn.Conv1d(in_channels=1200, out_channels=1200, kernel_size=3),
            nn.Dropout(0.2),
            nn.ReLU(),  # 7
            nn.Conv1d(in_channels=1200, out_channels=1200, kernel_size=3),
            nn.Dropout(0.2),
            nn.ReLU(),  # 5
            nn.Conv1d(in_channels=1200, out_channels=600, kernel_size=3),
            nn.Dropout(0.1),
            nn.ReLU(),  # 3
            nn.Conv1d(in_channels=600, out_channels=nhid, kernel_size=3),
            nn.ReLU(),  # 1
        )
        self.regress_conv = nn.Linear(nhid, 1)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_features_num, ntoken),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(ntoken, ninp),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(ninp, ninp),
            nn.Dropout(0.2),
            nn.ReLU(),
        )

        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers

    def forward(self, input):
        # emb = self.drop(self.encoder(input))
        cov_logits = self.conv1d_relu_stack(input)
        cov_logits = cov_logits.view(cov_logits.shape[0], cov_logits.shape[1])
        regress_conv_out = self.regress_conv(cov_logits)
        logits = self.linear_relu_stack(input)
        logits = torch.cat((logits, input), 2)
        # logits = logits.view(1, len(logits), -1)
        output, hidden = self.rnn(logits)
        output = output[:, -1, :]
        regress_rnn_out = self.regress_rnn(output)
        new_logits = torch.cat((cov_logits, output), 1)
        attention_output = self.self_attention(new_logits)
        # output = self.drop(output)
        new_logits = torch.mul(new_logits, attention_output)
        decoded_out = self.decoder(new_logits)
        # decoded_2 = self.decoder_2(decoded_1)
        return regress_conv_out, regress_rnn_out, decoded_out

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))


# dataloader = DataLoader(transformed_dataset, batch_size=4,
#                         shuffle=True, num_workers=0)
def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))


def RMSPELoss(y_pred, y_true):
    return torch.sqrt(torch.mean(((y_true - y_pred) / y_true) ** 2)).clone()


def do_process(optiver_ds, full_seconds_in_bucket, trade__, time_id):
    optiver_ds_ = optiver_ds[optiver_ds['time_id'] == time_id]
    if optiver_ds_.size == 0:
        return None
    optiver_ds_ = pd.merge(full_seconds_in_bucket, optiver_ds_, how='left', on='seconds_in_bucket')
    optiver_ds_ = pd.merge(optiver_ds_, trade__[trade__['time_id'] == time_id], how='left',
                           on='seconds_in_bucket')
    # optiver_ds_.drop(skip_cols)
    optiver_ds_ = optiver_ds_.drop(['time_id_x', 'time_id_y', 'seconds_in_bucket'], axis=1).fillna(0)
    # optiver_ds_ = np.nan_to_num(optiver_ds_)
    # TODO 将每一列进行标准化
    for i in range(optiver_ds_.shape[1]):
        optiver_ds_[lambda df: df.columns[0]]
        if np.sum(optiver_ds_[lambda df: df.columns[i]]) != 0 and np.std(optiver_ds_[lambda df: df.columns[i]]) != 0:
            a = (optiver_ds_[lambda df: df.columns[i]] - np.mean(optiver_ds_[lambda df: df.columns[i]])) / np.std(optiver_ds_[lambda df: df.columns[i]])
            optiver_ds_[lambda df: df.columns[i]] = a
    return optiver_ds_


def process_train_bach(arg):
    # input_0 = []
    # target_0 = []
    stock_id = arg['stock_id']
    time_id = arg['time_id']
    optiver_ds = arg['optiver_ds']
    full_seconds_in_bucket = arg['full_seconds_in_bucket']
    trade_train_ = arg['trade_train_']
    optiver_ds_ = do_process(optiver_ds, full_seconds_in_bucket, trade_train_, time_id)
    row_id = str(stock_id) + '-' + time_id.astype(str)
    np_target = train_ds[train_ds['row_id'] == row_id]
    # 创建的目录
    path = f"{DATA_PATH}formated_data/{stock_id}/"
    optiver_ds_.to_parquet(f'{path}/{time_id}.parquet')
    np_target.to_parquet(f'{path}/{time_id}_target.parquet')
    return optiver_ds_, np_target


def process_test_bach(time_id, ARGS):
    optiver_ds = ARGS['optiver_ds']
    full_seconds_in_bucket = ARGS['full_seconds_in_bucket']
    trade_test_ = ARGS['trade_test_']
    optiver_ds_ = do_process(optiver_ds, full_seconds_in_bucket, trade_test_, time_id)

    return optiver_ds_


def train_bach(epoch):
    full_seconds_in_bucket = {'seconds_in_bucket': np.arange(600)}  # seconds_in_bucket最大是600，训练数据中不连续，这里将他们连起来
    full_seconds_in_bucket = pd.DataFrame(full_seconds_in_bucket)
    # lstmmodel.zero_grad()
    # pool = Pool(30)  # 创建进程池，最大进程数为 NTHREADS
    for stock_id, stock_fnmame in book_train_dict.items():
        print(stock_id)
        path = f"{DATA_PATH}formated_data/{stock_id}/"
        if not os.path.exists(path):
            os.makedirs(path)
        trade_train_parquet = trade_train_dict.get(stock_id)
        trade_train_ = pd.read_parquet(trade_train_parquet)
        book_train = pd.read_parquet(stock_fnmame)
        time_ids = book_train['time_id'].unique()
        # hidden = lstmmodel.init_hidden(1)
        _start = time.time()
        param = []
        for time_id in time_ids:
            ARGS_ = dict(optiver_ds=book_train, full_seconds_in_bucket=full_seconds_in_bucket,
                         trade_train_=trade_train_, stock_id=stock_id, time_id=time_id)
            param.append(ARGS_)
            # process_train_bach(ARGS_)
        with Pool(8) as p:
            p.map(process_train_bach, param)
        # input_0, target_0 = zip(*r)
        # print(time.time() - _start)

        # ARGS = dict(optiver_ds=optiver_ds, full_seconds_in_bucket=full_seconds_in_bucket, trade_train_=trade_train_,
        #             stock_id=stock_id)
        # input_0=[]
        # target_0=[]
        # for time_id in time_ids:
        #     input_0_, target_0_ = process_train_bach(time_id, ARGS)
        #     input_0.append(input_0_)
        #     target_0.append(target_0_)
        # input_0, target_0 = pool.apply(func=process_train_bach, args=(time_ids,optiver_ds,full_seconds_in_bucket,trade_train_,stock_id))
        # input_0, target_0 = process_train_bach(time_ids,optiver_ds,full_seconds_in_bucket,trade_train_,stock_id)

        print(time.time() - _start)


        # logging.debug(f'epoch = {epoch} , stock_id = {stock_id} , time_id = {time_id} , loss = {loss_2.item()}')
        # snapshot = tracemalloc.take_snapshot()
        # top_stats = snapshot.statistics('lineno')
        # print("[ Top 10 ]")
        # for stat in top_stats[:10]:
        #     print(stat)
        # logging.debug('========================================================')

    # 每一个epoch之后就测试一下验证集
    # with torch.no_grad():
    #     test()

    # idx = np.arange(np_train.shape[0])
    # train_idx, valid_idx = train_test_split(idx, shuffle=True, test_size=0.1, random_state=SEED)


def start_train():
    train_bach(0)


def predict():
    lstmmodel = torch.load_state_dict(torch.load('train_out/model_weights_24.pth'))
    full_seconds_in_bucket = {'seconds_in_bucket': np.arange(600)}
    full_seconds_in_bucket = pd.DataFrame(full_seconds_in_bucket)
    # lstmmodel.zero_grad()
    loss_all = []
    # pool = Pool(30)  # 创建进程池，最大进程数为 NTHREADS
    target = []
    for index, row in test_ds.iterrows():
        # print(row['stock_id'])
        stock_id = row['stock_id']
        trade_test_id = book_train_dict.get(stock_id)
        trade_test_ = pd.read_parquet(trade_test_id)
        optiver_ds = pd.read_parquet(book_test_dict.get(stock_id))
        time_id = row['time_id']
        ARGS = dict(optiver_ds=optiver_ds, full_seconds_in_bucket=full_seconds_in_bucket, trade_test_=trade_test_,
                    stock_id=stock_id)
        input_0 = process_test_bach(time_id, ARGS)
        if input_0 is None:
            target.append(0)
            continue
        input_0 = input_0[None, :, :]
        input_1 = torch.tensor(input_0, dtype=torch.float32, requires_grad=True).to(device)
        with torch.no_grad():
            output_2, _ = lstmmodel(input_1)
        target.append(output_2.item())
    test_ds['target'] = target
    # print(test_ds)
    test_ds[['row_id', 'target']].to_csv('submission.csv', index=False)


if __name__ == '__main__':
    logging.debug('-------- start -----------')
    # print("CPU的核数为：{}".format(cpu_count()))

    NTHREADS = cpu_count()
    SEED = 42
    TRAIN_BATCH_SIZE = 3
    TEST_BATCH_SIZE = 256
    EPOCH_ACCOUNT = 250

    # DATA_PATH = '../input/optiver-realized-volatility-prediction'
    DATA_PATH = '/home/data/optiver-realized-volatility-prediction/'
    # DATA_PATH = '/home/szu/liyu/data/optiver-realized-volatility-prediction/'
    BOOK_TRAIN_PATH = DATA_PATH + 'book_train.parquet'
    TRADE_TRAIN_PATH = DATA_PATH + 'trade_train.parquet'
    BOOK_TEST_PATH = DATA_PATH + 'book_test.parquet'
    TRADE_TEST_PATH = DATA_PATH + 'trade_test.parquet'
    CHECKPOINT = './model_checkpoint/model_01'


    train_ds = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
    test_ds = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))

    print(f'Train ds shape: {train_ds.shape}')
    print(f'Test ds shape: {test_ds.shape}')
    train_ds['row_id'] = train_ds['stock_id'].astype(str) + '-' + train_ds['time_id'].astype(str)
    book_train_dict = get_path_dict(BOOK_TRAIN_PATH, train_ds['stock_id'].unique())
    trade_train_dict = get_path_dict(TRADE_TRAIN_PATH, train_ds['stock_id'].unique())

    book_test_dict = get_path_dict(BOOK_TEST_PATH, test_ds['stock_id'].unique())
    trade_test_dict = get_path_dict(TRADE_TEST_PATH, test_ds['stock_id'].unique())

    book_skip_columns = trade_skip_columns = ['time_id', 'row_id', 'target']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(device)

    input_features_num = 11

    start_train()
    print('============= finish =============')
