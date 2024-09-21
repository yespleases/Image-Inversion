import numpy as np
import torch
from torch import nn
from torch.utils import data as Data
from scipy.io import loadmat
from glob import glob
import random
from sklearn.preprocessing import StandardScaler
from model import Bert, BertConfig
from tqdm import tqdm
import cv2


uv_dis = loadmat("./uv_distribution.mat")["uv_distribution"]
stand = StandardScaler()
uv_dis = stand.fit_transform(uv_dis)
uv_dis = np.transpose(uv_dis, (1, 0))
device = "cuda" if torch.cuda.is_available() else "cpu"

def read_data(train_size=0.85):
    tb_path = glob("./tb_x/*")[:10000]
    #tb_path = glob("./tb_x/*")[:30000]
    uv_path = glob("./uv/*")

    assert len(tb_path) == len(uv_path), "lens"
    lens = len(tb_path)
    index = [i for i in range(0, lens)]

    random.shuffle(index)

    new_tb = []
    new_uv = []

    for i in index:
        new_tb.append(tb_path[i])
        new_uv.append(uv_path[i])


    train_index = int(lens * train_size)

    train_tb = new_tb[:train_index]
    test_tb = new_tb[train_index:]

    train_uv = new_uv[:train_index]
    test_uv = new_uv[train_index:]

    return train_tb, test_tb, train_uv, test_uv


class TBUVData(Data.Dataset):
    def __init__(self, tb_path, uv_path):
        super(TBUVData, self).__init__()
        self.tb_path, self.uv_path = tb_path, uv_path


    def __getitem__(self, item):
        new_tbx = []
        tb_p, uv_p = self.tb_path[item], self.uv_path[item]

        tb_y = loadmat(tb_p)["Y"]
        uv_x = loadmat(uv_p)["visibility"][0]

        for d in uv_x:
            new_tbx.append([d.real, d.imag])

        uv_x = np.float32(new_tbx)

        uv_x = np.concatenate([uv_x, uv_dis], axis=-1)
        tb_y = tb_y[:, :] / 255.0

        return np.float32(uv_x), np.float32(tb_y)



    def __len__(self):
        return len(self.tb_path)



def train():
    config = BertConfig.from_json_file("config.json")

    train_tb, test_tb, train_uv, test_uv = read_data()

    test_data = TBUVData(test_tb, test_uv)
    test_data = Data.DataLoader(test_data, shuffle=True, batch_size=1)

    model = Bert(config)
    model.load_state_dict(torch.load("model.pkl"))
    model.to(device)

    model.eval()  # 开启验证模式
    pbar = tqdm(test_data)
    # loss_all = None

    for step, (x, y) in enumerate(pbar):

        x = x.to(device)
        out = model(x)

        pred_img = out.cpu().detach().numpy()[0]

        cv2.imshow("a", pred_img)
        cv2.imshow("b", y.numpy()[0])
        cv2.waitKey(0)


if __name__ == '__main__':
    train()
