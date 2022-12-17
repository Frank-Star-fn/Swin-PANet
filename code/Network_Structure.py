import touch
import torch.nn as nn

n_channels = 3
n_labels = 1
epochs = 300
img_size = 224
vis_frequency = 10
early_stopping_patience = 100
pretrain = False
task_name = 'GlaS' # GlaS, MoNuSeg, and ISIC 2016
learning_rate = 5e-3
batch_size = 2
Transformer_patch_sizes = [2,4]
# Transformer.dropout_rate = 0.1

base_channel = 32
class Swin_PANet(nn.Module):
    def __init__(self, config, n_channels = 3, n_classes = 1,img_size = 224,vis = False):
        super().__init__()
        self.vis = vis
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = config.base_channel
        self.down1 = DownBlock(in_channels, in_channels*2, nb_Conv = 2)
        self.down2 = DownBlock(in_channels*2, in_channels*4, nb_Conv = 2)
        self.down3 = DownBlock(in_channels*4, in_channels*8, nb_Conv = 2)
        self.down4 = DownBlock(in_channels*8, in_channels*8, nb_Conv = 2)
        self.prior_attention_network = Intermediate_supervision(in_channels*2, in_channels*4, in_channels*8, in_channels*8)
        self.up4 = UpBlock(in_channels*16, in_channels*4, in_channels*4, nb_Conv = 2)
        self.up3 = UpBlock(in_channels*8, in_channels*2, in_channels*4, nb_Conv = 2)
        self.up2 = UpBlock(in_channels*4, in_channels, in_channels*4, nb_Conv = 2)
        self.up1 = UpBlock(in_channels*2, in_channels, in_channels*4, nb_Conv = 2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size = (1,1), stride = (1,1))
        self.last_activation = nn.Sigmoid() # if using BCELoss
    def forward(self, x):
        x = x.float()
        x1 = self.ConvBatchNorm(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        attention_prediction = self.prior_attention_network(x1,x2,x3,x4)
        x = self.up4(x5, x4, attention_prediction)
        x = self.up3(x, x3, attention_prediction)
        x = self.up2(x, x2, attention_prediction)
        x = self.up1(x, x1, attention_prediction)
        logits = self.last_activation(self.outc(x))
        return attention_prediction, logits

'''
model=Swin_PANet(config)
images, masks = batch['image'], batch['label']
images, masks = images.cuda(), masks.cuda()
preds, attention_prediction = model(images)
intermediate_loss = intermediate_criterion(attention_prediction, masks.float())
out_loss = direct_criterion(preds, masks.float())
intermediate_loss.backward(grad = True)
out_loss.backward()
optimizer.step()
'''

print('This is a brief introduction to the structure of the Swin-PANet.')
