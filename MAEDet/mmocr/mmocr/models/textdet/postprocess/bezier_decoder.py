import torch
import numpy as np


class BezierDecoder:


    def __init__(self,num_fiducial, n):
        self.n = n
        self.num_fiducial = num_fiducial

    def beizer_to_poly(self,cpts, t):
        # t = torch.from_numpy(np.linspace(0,1,self.n)).to(device)
        x0, y0, x1, y1, x2, y2, x3, y3 = torch.split(cpts,1,dim=1)
        bezier_x = (1 - t) * ((1 - t) * ((1 - t) * x0 + t * x1) + t * ((1 - t) * x1 + t * x2)) + t * (
                (1 - t) * ((1 - t) * x1 + t * x2) + t * ((1 - t) * x2 + t * x3))
        bezier_y = (1 - t) * ((1 - t) * ((1 - t) * y0 + t * y1) + t * ((1 - t) * y1 + t * y2)) + t * (
                (1 - t) * ((1 - t) * y1 + t * y2) + t * ((1 - t) * y2 + t * y3))
        # return np.stack([bezier_x, bezier_y], axis=1)
        return torch.stack([bezier_x, bezier_y], dim=-1)


    def tps2poly(self, cpts,direction=None):
        device = cpts.device
        batch_T = cpts.view(-1, self.num_fiducial*2)
        batch_size = batch_T.shape[0]

        t = torch.from_numpy(np.linspace(0,1,self.n,dtype='float32')).to(device)
        t = t[None].repeat(batch_size,1)
        top = self.beizer_to_poly(cpts[:, :self.num_fiducial],t)
        t = torch.from_numpy(np.linspace(0, 1, self.n,dtype='float32')).to(device)
        t = t[None].repeat(batch_size, 1)
        bot = self.beizer_to_poly(cpts[:, self.num_fiducial:], t)
        poly = torch.cat([top, bot], dim=1)
        return poly

