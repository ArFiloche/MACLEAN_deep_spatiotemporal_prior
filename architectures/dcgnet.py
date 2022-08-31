import torch.nn as nn

class DCGNet(nn.Module):
    def __init__(self, n_channel=32, n_z=100):
        super(DCGNet, self).__init__()
        
        self.name = 'DCGNet'
        
        # number of output channel
        self.n_channel=n_channel
        # size of 1d - white noise/latent space
        self.n_z=n_z
        self.main = nn.Sequential(
            unsqz(),
            
            nn.ConvTranspose2d(self.n_z, 256, 16, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),      
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64,kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, self.n_channel, kernel_size=3, stride=1, padding=0),

            sqz(),
        )

    def forward(self, input):
        return self.main(input)
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
class unsqz(nn.Module):
    def forward(self, input):
        return input.unsqueeze(0)
    
class sqz(nn.Module):
    def forward(self, input):
        return input.squeeze(0)