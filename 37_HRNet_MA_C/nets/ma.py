import torch.nn as nn

class ma_block(nn.Module):
    def __init__(self, Channel_H, Channel_L, Channel_T):
        super(ma_block, self).__init__()
        Channel_H = Channel_H
        Channel_L = Channel_L
        Channel_T = Channel_T

        self.q_h = nn.Linear(Channel_H, Channel_T)
        self.k_h = nn.Linear(Channel_H, Channel_T)
        self.v_h = nn.Linear(Channel_H, Channel_T)

        self.proj_h = nn.Linear(Channel_T, Channel_H)

        self.q_l = nn.Linear(Channel_L, Channel_T)
        self.k_l = nn.Linear(Channel_L, Channel_T)
        self.v_l = nn.Linear(Channel_L, Channel_T)

        self.proj_l = nn.Linear(Channel_T, Channel_L)

    def forward(self, x):
        B_h, C_h, H_h, W_h = x[0].shape
        B_l, C_l, H_l, W_l = x[1].shape
        x_h = x[0]
        x_l = x[1]

        xh_q = self.q_h(x_h.reshape(B_h, C_h, -1).transpose(-2, -1)).transpose(-2, -1)
        # xh_k = self.k_h(x_h.reshape(B_h, C_h, -1).transpose(-2, -1)).transpose(-2, -1)
        xh_v = self.v_h(x_h.reshape(B_h, C_h, -1).transpose(-2, -1)).transpose(-2, -1)
        # B, N, C1===>B, CC, N

        # xl_q = self.q_l(x_l.reshape(B_l, C_l, -1).transpose(-2, -1)).transpose(-2, -1)
        xl_k = self.k_l(x_l.reshape(B_l, C_l, -1).transpose(-2, -1)).transpose(-2, -1)
        xl_v = self.v_l(x_l.reshape(B_l, C_l, -1).transpose(-2, -1)).transpose(-2, -1)
        # B, N, C2===>B, CC, N

        QDKS = xh_q.transpose(-2, -1) @ xl_k
        QSKD = QDKS.transpose(-2, -1)
        # B, N, CC * B, CC, n===>B, N, n
        # QSKD = xl_q.transpose(-2, -1) @ xh_k

        QDKS = QDKS.softmax(dim=-1)
        QSKD = QSKD.softmax(dim=-1)

        AH = xh_v @ QDKS
        # B, CC, N * B, N, n===>B, CC, n
        AL = xl_v @ QSKD
        # B, CC, n * B, n, N == = > B, CC, N

        RL = self.proj_l(AH.transpose(-2, -1)).transpose(-2, -1).reshape(B_l, C_l, H_l, W_l)
        RH = self.proj_h(AL.transpose(-2, -1)).transpose(-2, -1).reshape(B_h, C_h, H_h, W_h)
        x_h = x_h + RH
        x_l = x_l + RL

        # print("x_h.shape")
        # print(x_h.shape)
        # print("xl.shape")
        # print(x_l.shape)
        # print("xh_q.shape")
        # print(xh_q.shape)
        # print("xl_q.shape")
        # print(xl_q.shape)
        # print("QDKS.shape")
        # print(QDKS.shape)
        # print("QSKD.shape")
        # print(QSKD.shape)
        # print("xh_v.shape")
        # print(xh_v.shape)
        # print("x1_q.shape")
        # print(xl_q.shape)
        # print("AH.shape")
        # print(AH.shape)
        # print("AL.shape")
        # print(AL.shape)
        # print("RL.shape")
        # print(RL.shape)
        # print("RH.shape")
        # print(RH.shape)
        # print("x_h.shape")
        # print(x_h.shape)
        # print("x_l.shape")
        # print(x_l.shape)
        return [x_h, x_l]
