import numpy as np
import torch
from torch import nn
from torch.utils import data as Data
from scipy.io import loadmat
from glob import glob
import random
from model import Bert, BertConfig
from tqdm import tqdm
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

def read_data(train_size=0.85):
    tb_path = glob("./tb_x/*")
    uv_path = glob("./uv_distribution/*")
    vs_path = glob("./vs/*")

    assert len(tb_path) == len(uv_path) == len(vs_path), "lens"
    lens = len(tb_path)
    index = [i for i in range(0, lens)]

    random.shuffle(index)

    new_tb = []
    new_uv = []
    new_vs = []

    for i in index:
        new_tb.append(tb_path[i])
        new_uv.append(uv_path[i])
        new_vs.append(vs_path[i])

    train_index = int(lens * train_size)

    train_tb = new_tb[:train_index]
    test_tb = new_tb[train_index:]

    train_uv = new_uv[:train_index]
    test_uv = new_uv[train_index:]

    train_vs = new_vs[:train_index]
    test_vs = new_vs[train_index:]

    return train_tb, test_tb, train_uv, test_uv, train_vs, test_vs


class TBUVData(Data.Dataset):
    def __init__(self, tb_path, uv_path, vs_path):
        super(TBUVData, self).__init__()
        self.tb_path, self.uv_path, self.vs_path = tb_path, uv_path, vs_path


    def __getitem__(self, item):
        new_tbx = []
        tb_p, uv_p, vs_p = self.tb_path[item], self.uv_path[item], self.vs_path[item]

        tb_y = loadmat(tb_p)["Y"]
        uv_x = loadmat(vs_p)["visibility"][0]
        # print(uv_x)
        vs_x = loadmat(uv_p)["uv_distribution"]
        vs_x = np.transpose(vs_x, (1, 0))


        for d in uv_x:
            new_tbx.append([d.real, d.imag])

        uv_x = np.float32(new_tbx)

        uv_x = np.concatenate([uv_x, vs_x], axis=-1)
        tb_y = tb_y[:, :] / 255.0

        return np.float32(uv_x), np.float32(tb_y), uv_x.shape[0]

    def __len__(self):
        return len(self.tb_path)

    def call_fc(self, batch):

        new_x = []
        new_y = []

        max_len = 465   #######################################
        
        #mask = np.float32([[-1e6]])

        #mask = np.repeat(mask, 4, axis=-1)
        #mask = np.repeat(mask, max_len, axis=0)

        #mask = [np.float32([[-1e6]]),np.float32([[-1e6]]),0,0]

        #mask = np.repeat(mask, max_len, axis=1)
        #mask = np.transpose(mask)

        mask = np.zeros((max_len,4))
        mask[:][3:4] = np.float32([[-1e6]])
        #mask[:][3:4] = np.float32([[-1000]])
   

        for x, y, i in batch:
            x_mask = mask
            x_mask[:i, :] = x

            new_x.append(x_mask)
            new_y.append(y)

        x = torch.FloatTensor(new_x)
        y = torch.FloatTensor(new_y)

        return x, y


def train():
    config = BertConfig.from_json_file("config.json")

    train_tb, test_tb, train_uv, test_uv, train_vs, test_vs = read_data()

    train_data = TBUVData(train_tb, train_uv, train_vs)
    max_ls = []
    # for _, _, i in train_data:
    #     print(i)
    #     max_ls.append(i)
    train_data = Data.DataLoader(train_data, shuffle=True, batch_size=config.batch_size, collate_fn=train_data.call_fc)

    test_data = TBUVData(test_tb, test_uv, test_vs)
    # for _, _, i in test_data:
    #     print(i)
    #     max_ls.append(i)
    # print(max(max_ls))
    # exit()
    test_data = Data.DataLoader(test_data, shuffle=True, batch_size=config.batch_size, collate_fn=test_data.call_fc)

    model = Bert(config)
    if os.path.exists("model.pkl"):
        model.load_state_dict(torch.load("model.pkl"))
    model.train()  # 开启训练模式
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)  # 初始化优化器
    loss_fc = nn.MSELoss()  # 创建损失函数

    test_loss = []
    train_loss = []

    loss_old = 10000

    nb = len(train_data)
    for epoch in range(1, config.epochs):
        pbar = tqdm(train_data, total=nb)
        loss_all = None
        for step, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)

            out = model(x)

            out = out * 10
            y = y * 10

            loss = loss_fc(out, y)

            loss.backward()  # 反向传播
            optimizer.step()  # 梯度下降
            optimizer.zero_grad()  # 清空梯度

            if loss_all is None:
                loss_all = loss.item()
                loss_time = loss_all
            else:
                loss_all += loss.item()
                loss_time = loss_all / (step + 1)

            s = ("train => epoch:{} - step:{} - loss:{:.3f} - loss_time:{:.3f}".format(epoch, step, loss, loss_time))
            pbar.set_description(s)
        train_loss.append(loss_time)
        model.eval()  # 开启验证模式
        pbar = tqdm(test_data)
        loss_all = None

        for step, (x, y) in enumerate(pbar):

            x, y = x.to(device), y.to(device)
            out = model(x)

            out = out * 10
            y = y * 10

            loss = loss_fc(out, y)

            if loss_all is None:
                loss_all = loss.item()
                test_loss_time = loss_all
            else:
                loss_all += loss.item()
                test_loss_time = loss_all / (step + 1)

            s = (
                "test => epoch:{} - step:{} - loss:{:.3f} - loss_time:{:.3f}".format(epoch, step, loss, test_loss_time))
            pbar.set_description(s)
        test_loss.append(test_loss_time)
        model.train()

        if loss_time < loss_old:
            loss_old = loss_time
            torch.save(model.state_dict(), "model.pkl")

if __name__ == '__main__':
    train()
