import torch
import torch.nn as nn

# define the CNN


class my_CNN(nn.Module):

    def __init__(self, pretrained_weights: str = None):
        super(my_CNN, self).__init__()

        # nn.Conv2d(input_channel,output_channel,kernel_size,stride,padding)
        conv1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)  # (64,512,32,32)
        pool1 = nn.MaxPool2d(2)                         # (64,512,16,16)
        bn1 = nn.BatchNorm2d(512)                       # (64,512,16,16)
        conv2 = nn.Conv2d(512, 128, 3, stride=1, padding=1)  # (64,128,16,16)
        pool2 = nn.MaxPool2d(2)                         # (64,128,8,8)
        bn2 = nn.BatchNorm2d(128)                       # (64,128,8,8)
        conv3 = nn.Conv2d(128, 64, 5, stride=1, padding=0)  # (64,64,4,4)
        bn3 = nn.BatchNorm2d(64)

        self.conv_part = nn.Sequential(
            conv1,
            bn1,
            nn.ReLU(),
            pool1,
            conv2,
            bn2,
            nn.ReLU(),
            pool2,
            conv3,
            bn3,
            nn.ReLU()
        )

        fc1 = nn.Linear(64*4*4, 512)
        bn4 = nn.BatchNorm1d(512)
        fc2 = nn.Linear(512, 128)
        bn5 = nn.BatchNorm1d(128)
        fc3 = nn.Linear(128, 2)

        self.fc_part = nn.Sequential(
            fc1,
            bn4,
            nn.ReLU(),
            fc2,
            bn5,
            nn.ReLU(),
            fc3,
            nn.Softmax(dim=1)
        )

        if pretrained_weights is not None:
            state_dict = torch.load(pretrained_weights, map_location=torch.device('cpu'))
            renamed_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.load_state_dict(renamed_state_dict)
            print("Load pretrained weights from {}".format(pretrained_weights))
    
    @torch.no_grad()
    def forward(self, x):
        out1 = self.conv_part(x)
        # dim = out1.shape[1] * out1.shape[2] * out1.shape[3]
        # out1 = out1.reshape(-1, dim)
        out1 = out1.reshape(out1.shape[0], -1)
        output = self.fc_part(out1)
        return output
