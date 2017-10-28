import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class GoogleNet(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # In order to keep input and output image sizes the same
        # 1x1 convolution kernel_size=1, padding=0
        # 3x3 convolution kernel_size=3, padding=1
        # 5x5 convolution kernel_size=5, padding=2
        #Google net 3a 3b inception module
        self.conv1_1_3a = nn.Conv2d(1, 64, 1,padding=0)
        self.conv3_3_reduce_3a=nn.Conv2d(1,96,1,padding=0)
        self.conv5_5_reduce_3a=nn.Conv2d(1,16,1,padding=0)
        self.conv3_3_3a=nn.Conv2d(96,128,3,padding=1)
        self.conv5_5_3a=nn.Conv2d(16,32,5,padding=2)
        self.conv1_1_max_pool_3a=nn.Conv2d(1,32,1,padding=0)

        self.conv1_1_3b = nn.Conv2d(256, 128, 1,padding=0)
        self.conv3_3_reduce_3b=nn.Conv2d(256,128,1,padding=0)
        self.conv5_5_reduce_3b=nn.Conv2d(256,32,1,padding=0)
        self.conv3_3_3b=nn.Conv2d(128,192,3,padding=1)
        self.conv5_5_3b=nn.Conv2d(32,96,5,padding=2)
        self.conv1_1_max_pool_3b=nn.Conv2d(256,64,1,padding=0)

        #google net 4a inception module
        self.conv1_1_4a = nn.Conv2d(480, 192, 1,padding=0)
        self.conv3_3_reduce_4a=nn.Conv2d(480,96,1,padding=0)
        self.conv5_5_reduce_4a=nn.Conv2d(480,16,1,padding=0)
        self.conv3_3_4a=nn.Conv2d(96,208,3,padding=1)
        self.conv5_5_4a=nn.Conv2d(16,48,5,padding=2)
        self.conv1_1_max_pool_4a=nn.Conv2d(480,64,1,padding=0)

        self.avg_pool_out_4a=nn.avgPool2d(5,stride=3);
        self.conv1_1_avg_pool_4a=nn.Conv2d(1,512,padding=0)
        self.fc1_4a=nn.Linear(4*4*512,1024);
        self.fc2_4a=nn.Dropout(p=0.7);
        self.fc3_4a=nn.Linear(1024,5);
        self.fc4_4a=nn.Softmax();
        '''self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)'''

    def forward(self, x):
        # Max pooling over a (2, 2) window
        conv1_1_3a=F.relu(self.conv1_1_3a(x))
        conv3_3_3a=F.relu(self.conv3_3_3a(F.relu(self.conv3_3_reduce_3a(x))))
        conv5_5_3a=F.relu(self.conv5_5_3a(F.relu(self.conv5_5_reduce_3a(x))))
        max_pool_3a=F.relu(self.conv1_1_max_pool_3a(F.relu(F.max_pool2d(x),3,padding=1)))
        3a_output=torch.cat((conv1_1_3a,conv3_3_3a,conv5_5_3a,max_pool_3a),0);

        conv1_1_3b=F.relu(self.conv1_1_3b(3a_output))
        conv3_3_3b=F.relu(self.conv3_3_3b(F.relu(self.conv3_3_reduce_3b(3a_output))))
        conv5_5_3b=F.relu(self.conv5_5_3b(F.relu(self.conv5_5_reduce_3b(3a_output))))
        max_pool_3b=F.relu(self.conv1_1_max_pool_3b(F.relu(F.max_pool2d(3a_output),3,padding=1)))
        3b_output=torch.cat((conv1_1_3b,conv3_3_3b,conv5_5_3b,max_pool_3b),0);   

        4a_input=F.relu(F.max_pool2d(3b_output),3,stride=2);
        conv1_1_4a=F.relu(self.conv1_1_3a(4a_input))
        conv3_3_4a=F.relu(self.conv3_3_3a(F.relu(self.conv3_3_reduce_3a(4a_input))))
        conv5_5_4a=F.relu(self.conv5_5_3a(F.relu(self.conv5_5_reduce_3a(4a_input))))
        max_pool_4a=F.relu(self.conv1_1_max_pool_3a(F.relu(F.max_pool2d(4a_input),3,padding=1)))
        4a_output=torch.cat((conv1_1_4a,conv3_3_4a,conv5_5_4a,max_pool_4a),0);
        
        4a_avg_output=F.relu(self.conv1_1_avg_pool_4a(F.relu(self.avg_pool_out_4a(4a_output))));
        4a_flatten=4a_avg_output.view(-1,self.num_flat_features(4a_avg_output));
        4a_flatten=F.relu(self.fc1_4a(4a_flatten))
        4a_flatten=F.relu(self.fc2_4a(4a_flatten))
        4a_flatten=F.relu(self.fc3_4a(4a_flatten))
        4a_flatten=self.fc4_4a(4a_flatten)
        return 4a_flatten;
        '''x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x'''

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    