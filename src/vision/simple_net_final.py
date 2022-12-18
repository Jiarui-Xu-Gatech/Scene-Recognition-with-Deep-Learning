import torch
import torch.nn as nn


class SimpleNetFinal(nn.Module):
    def __init__(self):
        """
        Init function to define the layers and loss function

        Note: Use 'mean' reduction in the loss_criterion. Read Pytorch documention to understand what it means
        """
        super().__init__()

        self.conv_layers = nn.Sequential()
        self.fc_layers = nn.Sequential()
        self.loss_criterion = None

        ############################################################################
        # Student code begin
        ############################################################################
        '''
        self.layer1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.layer2 = nn.MaxPool2d(kernel_size=3)
        self.layer3 = nn.ReLU()
        self.layer4 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.layer5 = nn.MaxPool2d(kernel_size=3)
        self.layer6 = nn.ReLU()
        self.layer7 = nn.Flatten()
        self.layer8 = nn.Linear(500, 100)
        self.layer9 = nn.Linear(100, 15)
        self.conv_layers = nn.Sequential(self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6)
        self.fc_layers = nn.Sequential(self.layer7, self.layer8, self.layer9)
        self.loss_criterion = nn.CrossEntropyLoss()
        '''


        



        self.layer1=nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.layer2=nn.BatchNorm2d(10)
        self.layer3=nn.MaxPool2d(kernel_size=2)
        self.layer4=nn.ReLU()
        self.layer5=nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.layer6=nn.BatchNorm2d(20)
        self.layer7=nn.MaxPool2d(kernel_size=2)
        self.layer8=nn.ReLU()
        self.layer9=nn.Dropout(p=0.5)
        self.layer10=nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5)
        self.layer11=nn.MaxPool2d(kernel_size=2)
        self.layer12=nn.ReLU()
        self.layer13=nn.Flatten()
        self.layer14=nn.Linear(800, 100)
        self.layer15=nn.Linear(100, 15)
        self.conv_layers = nn.Sequential(self.layer1,self.layer2,self.layer3,self.layer4,self.layer5,self.layer6,
                                         self.layer7,self.layer8,self.layer9,self.layer10,self.layer11,self.layer12)

        self.fc_layers = nn.Sequential(self.layer13,self.layer14,self.layer15)
        self.loss_criterion = nn.CrossEntropyLoss(reduction='mean')

        ############################################################################
        # Student code end
        ############################################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass with the net

        Args:
        -   x: the input image [Dim: (N,C,H,W)]
        Returns:
        -   y: the output (raw scores) of the net [Dim: (N,15)]
        """
        model_output = None
        ############################################################################
        # Student code begin
        ############################################################################

        x=self.conv_layers(x)
        model_output = self.fc_layers(x)
        
        ############################################################################
        # Student code end
        ############################################################################

        return model_output
