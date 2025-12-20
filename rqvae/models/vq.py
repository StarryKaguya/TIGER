import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import kmeans, sinkhorn_algorithm


class VectorQuantizer(nn.Module):

    def __init__(self, n_e, e_dim,
                 beta = 0.25, kmeans_init = False, kmeans_iters = 10,
                 sk_epsilon=0.003, sk_iters=100,):
        super().__init__()
        self.n_e = n_e # 聚类数量
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters # kmeans 最大迭代次数
        self.sk_epsilon = sk_epsilon # sinkhorn epsilon
        self.sk_iters = sk_iters # 50

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        if not kmeans_init:
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.initted = False
            self.embedding.weight.data.zero_()

    def get_codebook(self):
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape=None):
        # get quantized latent vectors
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)

        return z_q

    def init_emb(self, data):

        centers = kmeans(
            data,
            self.n_e,
            self.kmeans_iters,
        )

        self.embedding.weight.data.copy_(centers)
        self.initted = True

    @staticmethod
    def center_distance_for_constraint(distances):
        # distances: B, K
        # distances: [B, K] (B个样本，K个聚类中心)
        # 将距离归一化到大致 [-1, 1] 的范围
        # Sinkhorn 算法涉及指数运算 exp(-dist/epsilon)。如果距离数值过大或差异过剧烈，会导致指数结果溢出（inf）或下溢（0），
        # 造成数值不稳定。通过归一化，保证输入数值在合理范围内，提高算法的鲁棒性。
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances

    def forward(self, x, use_sk=True):
        # Flatten input
        latent = x.view(-1, self.e_dim)

        if not self.initted and self.training:
            self.init_emb(latent)

        # Calculate the L2 Norm between latent and Embedded weights
        # 这几行代码利用了数学公式 $(a - b)^2 = a^2 + b^2 - 2ab$ 的展开形式，来避免直接计算 $(N, 1, D) - (1, K, D)$ 这种高显存消耗的广播操作。
        d = torch.sum(latent**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()- \
            2 * torch.matmul(latent, self.embedding.weight.t()) # (Batch_Size, Num_Embeddings)
        if not use_sk or self.sk_epsilon <= 0:
            indices = torch.argmin(d, dim=-1) # (B,) 容易导致 Codebook Collapse（码本坍塌） 问题
        else:
            d = self.center_distance_for_constraint(d)
            d = d.double()
            Q = sinkhorn_algorithm(d, self.sk_epsilon, self.sk_iters) # Q: (B, K)

            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print(f"Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.argmax(Q, dim=-1) # (B,)

        # indices = torch.argmin(d, dim=-1)

        x_q = self.embedding(indices).view(x.shape)

        # compute loss for embedding
        # 把loss拆开：如果你让同一个误差同时更新 encoder 与 codebook，它们会在离散边界附近“追逐振荡”
        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())
        loss = codebook_loss + self.beta * commitment_loss  # 我们通常认为 “更新 Codebook 让它适应数据” (Codebook Loss) 比 “限制 Encoder 的灵活性” (Commitment Loss) 更重要（或者说收敛速度不同）。

        # preserve gradients
        x_q = x + (x_q - x).detach() # STE的功能：Decoder 传回来的梯度，直接跳过了不可导的量化层，原封不动地复制给了 Encoder 的输出 x。

        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices


