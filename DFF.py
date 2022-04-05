# This code is modified from https://github.com/pkumivision/FFC
import torch
import torch.nn as nn
class Filtering(nn.Module):
    maml = False #Default
    def __init__(self, in_channels, out_channels):
        super(Filtering, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=1,
                                        kernel_size=7, stride=1, padding=3, bias=False)

    def forward(self, x):
        batch, _, _, _ = x.size()
        output_1 = self.relu(self.bn(self.conv1(x)))
            
        max_pool = torch.max(output_1,dim=1)[0].view((batch, 1,)+ (output_1.size()[2:]))
        avg_pool = torch.mean(output_1,dim=1).view((batch, 1,)+ (output_1.size()[2:]))

        output_2 = torch.cat((max_pool,avg_pool),dim=1)
        output_2 = torch.sigmoid(self.conv2(output_2))

        output = torch.mul(output_1, output_2)
        return output

class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups

        self.Filtering=Filtering(in_channels=in_channels*2, out_channels=out_channels*2)

    def forward(self, x):
        batch, c, h, w = x.size()
        ffted = torch.fft.rfftn(x, dim=(-2,-1), norm='ortho')   # (batch, c, h, w/2+1)
        ffted = torch.cat((ffted.real, ffted.imag), dim=1)   # (batch, c, h, w/2+1, 2)
        
        ffted = self.Filtering(ffted)   # (batch, c*2, h, w/2+1)
        
        ffted = torch.complex(ffted[:,:c,:,:], ffted[:,c:,:,:])  # (batch, c, h, w/2+1, 2)
        output = torch.fft.irfftn(ffted, dim=(-2,-1),norm='ortho')   # (batch, c, h, w)

        return output