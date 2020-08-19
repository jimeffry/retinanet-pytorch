import sys
import os
import torch.nn as nn

class MaskPredictor(nn.Module):
    """
    Heads for FPN for classification
    """
    def __init__(self,input_size,representation_size,num_classes):
        """
        Arguments:
            num_classes (int): number of output classes
            input_size (int): number of channels of the input once it's flattened
            representation_size (int): size of the intermediate representation
        """
        super(MaskPredictor, self).__init__()
        insize = representation_size
        self.conv1 = nn.Conv2d(input_size, insize, kernel_size=3, stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(insize,insize,3,stride=1,padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(insize,insize,3,stride=1,padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(insize,insize,3,stride=1,padding=1)
        self.act4 = nn.ReLU()
        self.upsample = nn.ConvTranspose(insize,insize,2,stride=2,padding=0)
        self.act5 = nn.ReLU()
        self.maskout = nn.Conv2d(insize,num_classes,1,1,0)
        self.__weight_init()
        
    def __weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            
    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.conv4(x)
        x = self.act4(x)
        x = self.upsample(x)
        x = self.act5(x)
        x = self.maskout(x)
        return x