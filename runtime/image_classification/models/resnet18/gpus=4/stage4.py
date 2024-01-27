import torch


class Stage4(torch.nn.Module):
    def __init__(self):
        super(Stage4, self).__init__()
        self.layer1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer3 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer4 = torch.nn.ReLU()
        self.layer5 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer6 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer8 = torch.nn.ReLU()
        self.layer9 = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.layer10 = torch.nn.Flatten(start_dim=1, end_dim=-1)
        self.layer11 = torch.nn.Linear(in_features=512, out_features=1000, bias=True)

    

    def forward(self, input0):
        out0 = input0.clone()
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out6 = out6 + out1
        out8 = self.layer8(out6)
        out9 = self.layer9(out8)
        out10 = self.layer10(out9)
        out11 = self.layer11(out10)
        return out11
