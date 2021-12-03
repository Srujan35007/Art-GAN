import torch 
import torch.nn as nn 


# Generator
class GeneratorBlock(nn.Module):
    def __init__(self, ins, outs, ksize, stride, pad):
        super(GeneratorBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(ins, outs, ksize, stride, pad),
            nn.BatchNorm2d(outs),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, block_x):
        return self.block(block_x)

class Generator(nn.Module):
    def __init__(self, img_channels=3, z_dims=1024):
        init_dims = 256
        super(Generator, self).__init__()
        # Input shape: (-1, z_dims, 1, 1)
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(z_dims, init_dims//1, 4, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),

            GeneratorBlock(init_dims//1, init_dims//2, 4, 2, 1),
            GeneratorBlock(init_dims//2, init_dims//4, 4, 2, 1),
            GeneratorBlock(init_dims//4, init_dims//8, 4, 2, 1),
            GeneratorBlock(init_dims//8, init_dims//16, 4, 2, 1),
            GeneratorBlock(init_dims//16, init_dims//32, 4, 2, 1),
            GeneratorBlock(init_dims//32, init_dims//64, 4, 2, 1),

            nn.ConvTranspose2d(init_dims//64, img_channels, 4, 2, 1),
            nn.Tanh(),
        )
        self._init_weights()

    def forward(self, x):
        return self.layers(x)

    def _init_weights(self):
        total, init = 0, 0
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                nn.init.normal_(module.weight.data, 0, 0.02)
                init += 1
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.normal_(module.weight.data, 0, 0.02)
                init += 1
            total += 1
        print(f"{self.__class__} model weights initialized {init}/{total}")
    

# Discriminator
class DiscriminatorBlock(nn.Module):
    def __init__(self, ins, outs, ksize, stride, pad):
        super(DiscriminatorBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ins, outs, ksize, stride, pad),
            nn.BatchNorm2d(outs),
            nn.ReLU(inplace=True),
        )

    def forward(self, block_x):
        return self.block(block_x)

class Discriminator(nn.Module):
    def __init__(self, img_channels=3):
        super(Discriminator, self).__init__()
        init_dims = 8
        self.layers = nn.Sequential(
            nn.Conv2d(img_channels, init_dims*1, 4, 2, 1),
            nn.ReLU(inplace=True),

            DiscriminatorBlock(init_dims*1, init_dims*2, 4, 2, 1),
            DiscriminatorBlock(init_dims*2, init_dims*4, 4, 2, 1),
            DiscriminatorBlock(init_dims*4, init_dims*8, 4, 2, 1),
            DiscriminatorBlock(init_dims*8, init_dims*16, 4, 2, 1),
            DiscriminatorBlock(init_dims*16, init_dims*32, 4, 2, 1),
            DiscriminatorBlock(init_dims*32, init_dims*64, 4, 2, 1),

            nn.Conv2d(init_dims*64, 1, 4, 1, 0),
            nn.Sigmoid(),
        )
        self._init_weights()

    def forward(self, x):
        return self.layers(x)
    
    def _init_weights(self):
        total, init = 0, 0
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                nn.init.normal_(module.weight.data, 0, 0.02)
                init += 1
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.normal_(module.weight.data, 0, 0.02)
                init += 1
            total += 1
        print(f"{self.__class__} model weights initialized {init}/{total}")

#from torchsummary import summary
#disc = Discriminator()
#gen = Generator()
#summary(disc, (3,512,512))
#summary(gen, (1024,1,1))
