system_content = "You are a expert in the field of neural architecture search."

user_input = '''Your task is to assist me in selecting the best channel numbers for a given model architecture. The model will be trained and tested on CIFAR10, and your objective will be to maximize the model's performance on CIFAR10. 

The model architecture will be defined as the following.
{
    layer1: nn.Conv2d(in_channels=3, out_channels=channels[0], kernel_size=3, padding=1, bias=False),
    layer2: InvertedResidual(in_channels=channels[0], bottleneck_channels=channels[1], out_channels=channels[0], stride=1),
    layer3: InvertedResidual(in_channels=channels[0], bottleneck_channels=channels[2], out_channels=channels[0], stride=1),
    layer4: InvertedResidual(in_channels=channels[0], bottleneck_channels=channels[3], out_channels=channels[4], stride=2),
    layer5: InvertedResidual(in_channels=channels[4], bottleneck_channels=channels[5], out_channels=channels[4], stride=1),
    layer6: nn.Conv2d(channels[4], channels[6], kernel_size=1, stride = 1, padding=0, bias=False),
    layer7: nn.AdaptiveAvgPool2d(output_size=1),
    layer8: nn.Linear(in_features=channels[6], out_features=10),
}

The implementation of the InvertedResidual is as follows:
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck_channels, stride):
        super(InvertedResidual, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, padding=1, groups=bottleneck_channels, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.use_shortcut = in_channels == out_channels and stride == 1

    def forward(self, x):
        if self.use_shortcut:
            return self.conv(x) + x
        return self.conv(x)

For the `channels` variable, the available channel number for each index would be:
{
    channels[0]: [32,  64,  96,  128],
    channels[1]: [192, 384, 576, 768],
    channels[2]: [192, 384, 576, 768],
    channels[3]: [192, 384, 576, 768],
    channels[4]: [64,  128, 192, 256],
    channels[5]: [384, 768, 1152, 1536],
    channels[6]: [256, 512, 768, 1024],
}

Your objective is to define the optimal number of channels for each layer based on the given options above to maximize the model's performance on CIFAR10. 
Your response should be the a channel list consisting of 7 numbers (e.g. [64, 576, ..., 256]).
'''

experiments_prompt = lambda arch_list, acc_list : '''Here are some experimental results that you can use as a reference:
{}
Please suggest a better channel list that can improve the model's performance on CIFAR10 beyond the experimental results provided above.
'''.format(''.join(['{} gives an accuracy of {:.2f}%\n'.format(arch, acc) for arch, acc in zip(arch_list, acc_list)]))

suffix = '''Please do not include anything else other than the channel list in your response.'''









system_content2 = "You are an expert in the field of neural architecture search."

user_input2 = '''Your task is to assist me in selecting the best channel numbers for a given model architecture. The model will be trained and tested on CIFAR10, and your objective will be to maximize the model's performance on CIFAR10. 

The model architecture will be defined as the following.
{
    layer1: nn.Conv2d(in_channels=3, out_channels=channels[0], kernel_size=3, padding=1, bias=False),
    layer2: BottleneckResidualBlock(in_channels=channels[0], bottleneck_channels=channels[1], out_channels=channels[0], stride=1),
    layer3: BottleneckResidualBlock(in_channels=channels[0], bottleneck_channels=channels[2], out_channels=channels[0], stride=1),
    layer4: BottleneckResidualBlock(in_channels=channels[0], bottleneck_channels=channels[3], out_channels=channels[4], stride=2),
    layer5: BottleneckResidualBlock(in_channels=channels[4], bottleneck_channels=channels[5], out_channels=channels[4], stride=1),
    layer6: BottleneckResidualBlock(in_channels=channels[4], bottleneck_channels=channels[6], out_channels=channels[4], stride=1),
    layer7: nn.AdaptiveAvgPool2d(output_size=1),
    layer8: nn.Linear(in_features=channels[4], out_features=10),
}

The implementation of the BottleneckResidualBlock is as follows:
class BottleneckResidualBlock(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, out_channels, stride):
        super().__init__()

        self.stride = stride

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, 3, stride = stride, padding=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, out_channels, 3, stride = 1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.stride == 1:
            return self.relu(x + self.block(x))
        else:
            return self.relu(self.block(x))

For the `channels` variable, the available channel number for each index would be:
{
    channels[0]: [64, 128, 192, 256],
    channels[1]: [64, 128, 192, 256],
    channels[2]: [64, 128, 192, 256],
    channels[3]: [128, 256, 384, 512],
    channels[4]: [128, 256, 384, 512],
    channels[5]: [128, 256, 384, 512],
    channels[6]: [128, 256, 384, 512],
}

Your objective is to define the optimal number of channels for each layer based on the given options above to maximize the model's performance on CIFAR10. 
Your response should be the a channel list consisting of 7 numbers (e.g. [64, 192, ..., 256]).
'''


experiments_prompt2 = lambda arch_list, acc_list : '''Here are some experimental results that you can use as a reference:
{}
Please suggest a channel list that can improve the model's performance on CIFAR10 beyond the experimental results provided above.
'''.format(''.join(['{} gives an accuracy of {:.2f}%\n'.format(arch, acc) for arch, acc in zip(arch_list, acc_list)]))


suffix2 = '''Please do not include anything else other than the channel list in your response.'''









system_content3 = "You are Quoc V. Le, a computer scientist and artificial intelligence researcher who is widely regarded as one of the leading experts in deep learning and neural network architecture search. Your work in this area has focused on developing efficient algorithms for searching the space of possible neural network architectures, with the goal of finding architectures that perform well on a given task while minimizing the computational cost of training and inference."

user_input3 = '''

You are Quoc V. Le, a computer scientist and artificial intelligence researcher who is widely regarded as one of the leading experts in deep learning and neural network architecture search. Your work in this area has focused on developing efficient algorithms for searching the space of possible neural network architectures, with the goal of finding architectures that perform well on a given task while minimizing the computational cost of training and inference.
You are an expert in the field of neural architecture search. 
Your task is to assist me in selecting the best operations to design a neural network 
The objective is to maximize the model's performance.

Your work in this area has focused on developing efficient algorithms for searching the 
space of possible neural network architectures, with the goal of finding architectures 
that perform well on a given task while minimizing the computational cost of training and inference.

Use the code interpreter tool, use it to execute code!

Let's break this down step by step:

Next, please consider the gradient flow based on the ideal model architecture.
For example, how the gradient from the later stage affects the earlier stage.
Now, answer the question - how we can design a high-performance model using the available operations?
Based the analysis, your task is to propose a model design with the given operations that prioritizes performance, without considering factors such as size and complexity.

After you suggest a design, I will test its actual performance and provide you with feedback. 
Based on the results of previous experiments, we can collaborate to iterate and improve the design. P
lease avoid suggesting the same design again during this iterative process.
'''

experiments_prompt3 = lambda x : '''By using this model, we achieved an accuracy of {}%. Please recommend a new model that outperforms prior architectures based on the abovementioned experiments. Also, Please provide a rationale explaining why the suggested model surpasses all previous architectures.'''.format(x)







system_content5 = "You are an expert in the field of neural architecture search."

user_input5 = '''Your task is to assist me in selecting the best operations for a given model architecture, which includes some undefined layers and available operations. The model will be trained and tested on CIFAR10, and your objective will be to maximize the model's performance on CIFAR10.

We define the 3 available operations as the following:
0: Identity(in_channels, out_channels, stride)
1: InvertedResidual(in_channels, out_channels, stride expansion=3, kernel_size=3)
2: InvertedResidual(in_channels, out_channels, stride expansion=6, kernel_size=5)

The implementation of the Identity is as follows:
class Identity(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Identity, self).__init__()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        return x

The implementation of the InvertedResidual is as follows:
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion, kernel_size):
        super(InvertedResidual, self).__init__()
        hidden_dim = in_channels * expansion
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.use_shortcut = in_channels == out_channels and stride == 1

    def forward(self, x):
        if self.use_shortcut:
            return self.conv(x) + x
        return self.conv(x)
        

The model architecture will be defined as the following.
{
    layer1:  {defined: True,  operation: nn.Conv2d(in_channels=3,  out_channels=32, kernel_size=3, padding=1, bias=False)},
    layer2:  {defined: False, downsample: True , in_channels: 32,  out_channels: 64 , stride: 2},
    layer3:  {defined: False, downsample: False, in_channels: 64,  out_channels: 64 , stride: 1},
    layer4:  {defined: False, downsample: True , in_channels: 64,  out_channels: 128, stride: 2},
    layer5:  {defined: False, downsample: False, in_channels: 128, out_channels: 128, stride: 1},
    layer6:  {defined: False, downsample: False, in_channels: 128, out_channels: 128, stride: 1},
    layer7:  {defined: False, downsample: True , in_channels: 128, out_channels: 256, stride: 2},
    layer8:  {defined: False, downsample: False, in_channels: 256, out_channels: 256, stride: 1},
    layer9:  {defined: False, downsample: False, in_channels: 256, out_channels: 256, stride: 1},
    layer10: {defined: True,  operation: nn.Conv2d(in_channels=256, out_channels=1280, kernel_size=1, bias=False, stride=1)},
    layer11: {defined: True,  operation: nn.AdaptiveAvgPool2d(output_size=1)},
    layer12: {defined: True,  operation: nn.Linear(in_features=1280, out_features=10)},
}

The currently undefined layers are layer2 - layer9, and the in_channels and out_channels have already been defined for each layer. To maximize the model's performance on CIFAR10, please provide me with your suggested operation for the undefined layers only. 

Your response should be an operation ID list for the undefined layers. For example:
[1, 2, ..., 0] means we use operation 1 for layer2, operation 2 for layer3, ..., operation 0 for layer9.
'''

experiments_prompt4 = lambda arch_list, acc_list : '''Here are some experimental results that you can use as a reference:
{}
Please suggest a better operation ID list that can improve the model's performance on CIFAR10 beyond the experimental results provided above.
'''.format(''.join(['{} gives an accuracy of {:.2f}%\n'.format(arch, acc) for arch, acc in zip(arch_list, acc_list)]))

suffix4 = '''Please do not include anything other than the operation ID list in your response.'''
