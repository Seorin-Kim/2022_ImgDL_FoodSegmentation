import torch
import torch.nn as nn
import ResidualBlock

class ResUNet(nn.Module):
    def __init__(self, num_classes):
        super(ResUNet, self).__init__()
        self.num_classes = num_classes
        
        """ Encoder input layer """
        self.contl_1 = self.input_block(in_channels=3, out_channels=64)
        self.contl_2 = self.input_skip(in_channels=3, out_channels=64)
        
        """ Residual encoder block """
        self.resdl_1 = ResidualBlock(64, 128, 2, 1)
        self.resdl_2 = ResidualBlock(128, 256, 2, 1)
        #self.resdl_3 = ResidualBlock(256, 512, 2, 1)
        
        """ Encoder decoder skip connection """
        self.middle = ResidualBlock(256, 512, 2, 1)
        
        """ Decoder block """
        self.expnl_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, 
                                          kernel_size=2, stride=2, padding=0)
        self.expnl_1_cv = ResidualBlock(256+256, 256, 1, 1)
        self.expnl_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, 
                                          kernel_size=2, stride=2, padding=0)
        self.expnl_2_cv = ResidualBlock(128+128, 128, 1, 1)
        self.expnl_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, 
                                          kernel_size=2, stride=2, padding=0)
        self.expnl_3_cv = ResidualBlock(64+64, 64, 1, 1)
        # self.expnl_4 = nn.ConvTranspose2d(in_channels=128, out_channels=128, 
        #                                   kernel_size=2, stride=2, padding=0)
        # self.expnl_4_cv = ResidualBlock(128+64, 64, 1, 1)
        
        self.output = nn.Sequential(
          nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1),
          nn.Sigmoid(),
        )
        
    def input_block(self, in_channels, out_channels):
        block = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(num_features=out_channels),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                    )
        return block
    
    def input_skip(self, in_channels, out_channels):
        skip = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        return skip                         
    
    def forward(self, X):
        contl_1_out = self.contl_1(X) # 64
        contl_2_out = self.contl_2(X) # 64
        input_out = contl_1_out + contl_2_out
        
        resdl_1_out = self.resdl_1(input_out) # 128
        resdl_2_out = self.resdl_2(resdl_1_out) # 256
        #resdl_3_out = self.resdl_3(resdl_2_out) # 512
        
        middle_out = self.middle(resdl_2_out) # 512
        
        expnl_1_out = self.expnl_1(middle_out)
        expnl_1_cv_out = self.expnl_1_cv(torch.cat((expnl_1_out, resdl_2_out), dim=1)) # 512
        expnl_2_out = self.expnl_2(expnl_1_cv_out) # 256
        expnl_2_cv_out = self.expnl_2_cv(torch.cat((expnl_2_out, resdl_1_out), dim=1))
        expnl_3_out = self.expnl_3(expnl_2_cv_out)
        expnl_3_cv_out = self.expnl_3_cv(torch.cat((expnl_3_out, contl_1_out), dim=1))
        # expnl_4_out = self.expnl_4(expnl_3_cv_out)
        # expnl_4_cv_out = self.expnl_4_cv(torch.cat((expnl_4_out, input_out), dim=1))
        
        out = self.output(expnl_3_cv_out)
        return out