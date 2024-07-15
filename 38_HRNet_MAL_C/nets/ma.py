import torch.nn as nn

class ma_block(nn.Module):
    def __init__(self, Channel_H, Channel_L, Channel_T):
        super(ma_block, self).__init__()
        Channel_H = Channel_H
        Channel_L = Channel_L
        Channel_T = Channel_T

        self.qk_h = nn.Linear(Channel_H, Channel_T)
        self.v_h  = nn.Linear(Channel_H, Channel_T)

        self.proj_h = nn.Linear(Channel_T, Channel_H)

        self.qk_l = nn.Linear(Channel_L, Channel_T)
        self.v_l  = nn.Linear(Channel_L, Channel_T)

        self.proj_l = nn.Linear(Channel_T, Channel_L)

    def forward(self, x):
        B_h, C_h, H_h, W_h = x[0].shape
        B_l, C_l, H_l, W_l = x[1].shape
        x_h = x[0]
        x_l = x[1]

        xh_qk = self.qk_h(x_h.reshape(B_h, C_h, -1).transpose(-2, -1)).transpose(-2, -1)
        # xh_v  = self.v_h(x_h.reshape(B_h, C_h, -1).transpose(-2, -1)).transpose(-2, -1)
        # B, N, C1===>B, N, CC ==> B, CC, N

        xl_qk = self.qk_l(x_l.reshape(B_l, C_l, -1).transpose(-2, -1)).transpose(-2, -1)
        xl_v  = self.v_l(x_l.reshape(B_l, C_l, -1).transpose(-2, -1)).transpose(-2, -1)
        # B, n, C2===>B, n, CC ==> B, CC, n

        QDKS = xh_qk.transpose(-2, -1) @ xl_qk
        QSKD = QDKS.transpose(-2, -1)
        # B, N, CC * B, CC, n===>B, N, n
        # B, N, n===>B, n, N

        # QDKS = QDKS.softmax(dim=-1)
        QSKD = QSKD.softmax(dim=-1)

        # AH = xh_v @ QDKS
        # B, CC, N * B, N, n===>B, CC, n
        AL = xl_v @ QSKD
        # B, CC, n * B, n, N == = > B, CC, N

        RH = self.proj_h(AL.transpose(-2, -1)).transpose(-2, -1).reshape(B_h, C_h, H_h, W_h)
        #B, CC, N===>B, N, CC===>B, N, C1===>B, C1, N===>B, C1, H1, W1
        # RL = self.proj_l(AH.transpose(-2, -1)).transpose(-2, -1).reshape(B_l, C_l, H_l, W_l)
        #B, CC, n===>B, n, CC===>B, n, C2===>B, C2, n===>B, C2, H2, W2

        x_h = x_h + RH
        # x_l = x_l + RL

        return x_h
