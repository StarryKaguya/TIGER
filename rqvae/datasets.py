import pandas as pd
import torch
import torch.utils.data as data
import numpy as np


class EmbDataset(data.Dataset):
    '''
    把硬盘里的 Parquet 文件加载到内存中，并“清洗”成 PyTorch 训练能直接使用的 Tensor 格式。
    '''
    def __init__(self,data_path):

        self.data_path = data_path
        self.embeddings = pd.read_parquet(data_path)['embedding'].values
        self.embeddings = np.stack(self.embeddings, axis=0)
        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb=torch.FloatTensor(emb)
        return tensor_emb

    def __len__(self):
        return len(self.embeddings)
