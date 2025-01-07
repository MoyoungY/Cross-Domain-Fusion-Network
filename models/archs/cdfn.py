import math
import torch
from torch import nn, optim
import torch.fft
from models.backbones.resunet import ResUnet, ResUnet_DS
from tools.propagation.ASM import propagation_ASM

class IPM(nn.Module):
    """computes the initial input phase given a target amplitude"""
    def __init__(self):
        super(IPM, self).__init__()

    def forward(self, angle_tensor):
        sin = torch.sin(angle_tensor)
        cos = torch.cos(angle_tensor)
        phase = torch.atan2(sin, cos)
        return phase


class InitialPhaseUnet(nn.Module):
    """computes the initial input phase given a target amplitude"""
    def __init__(self, is_IPM=False, is_DS=False):
        super(InitialPhaseUnet, self).__init__()

        self.is_IPM = is_IPM
        self.is_DS = is_DS

        if self.is_DS:
            self.net = ResUnet(1, 1, filters=[16, 32, 64, 128])
        else:
            self.net = ResUnet(1, 1, filters=[16, 32, 64, 128])

        if self.is_IPM:
            self.phase = IPM()
        else:
            self.phase = nn.Hardtanh(-math.pi, math.pi)

    def forward(self, amp):
        out = self.net(amp)
        out_phase = self.phase(out)
        return out_phase


class FinalPhaseOnlyUnet(nn.Module):
    """computes the final SLM phase given a naive SLM amplitude and phase"""
    def __init__(self, is_IPM=False, is_DS=False):
        super(FinalPhaseOnlyUnet, self).__init__()

        self.is_IPM = is_IPM
        self.is_DS = is_DS

        if self.is_DS:
            self.net = ResUnet_DS(2, 1, filters=[16, 32, 64, 128])
        else:
            self.net = ResUnet(2, 1, filters=[16, 32, 64, 128])

        if self.is_IPM:
            self.phase = IPM()
        else:
            self.phase = nn.Hardtanh(-math.pi, math.pi)

    def forward(self, amp_phase):
        out = self.net(amp_phase)
        if self.is_DS:
            return self.phase(out[0]), self.phase(out[1]), self.phase(out[2]), self.phase(out[3])
        else:
            return self.phase(out)


class CDFN(nn.Module):
    def __init__(self, is_IPM=False, is_DS=False):
        super().__init__()

        self.is_IPM = is_IPM
        self.is_DS = is_DS
        self.initial_phase = InitialPhaseUnet(is_IPM=self.is_IPM, is_DS=self.is_DS)
        self.final_phase_only = FinalPhaseOnlyUnet(is_IPM=self.is_IPM, is_DS=self.is_DS)

    def forward(self, amp, Hbackward=None):
        predict_phase = self.initial_phase(amp)
        predict_complex = amp * torch.exp(1j*predict_phase)
        slmfield = propagation_ASM(predict_complex, precomped_H=Hbackward)

        slm_amp_phase = torch.cat((slmfield.abs(), slmfield.angle()), -3)
        holophase = self.final_phase_only(slm_amp_phase)

        return holophase