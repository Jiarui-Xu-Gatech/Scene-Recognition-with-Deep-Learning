import torch
import torch.nn as nn
from torchvision.models import resnet18


class MultilabelResNet18(nn.Module):
    def __init__(self):
        """Initialize network layers.

        Note: Do not forget to freeze the layers of ResNet except the last one
        Note: Consider which activation function to use
        Note: Use 'mean' reduction in the loss_criterion. Read Pytorch documention to understand what it means

        Download pretrained resnet using pytorch's API (Hint: see the import statements)
        """
        super().__init__()

        self.conv_layers = None
        self.fc_layers = None
        self.loss_criterion = None
        self.activation = None

        ############################################################################
        # Student code begin
        ############################################################################
        self.activation=nn.Sigmoid()
        model = resnet18(pretrained=True)
        list_children = list(model.children())

        self.fc_layers = nn.Sequential(nn.Linear(512, 7))
        self.conv_layers = nn.Sequential(*(list_children)[:-1])

        idx_chidren = 0
        for child in self.conv_layers.children():

            idx_chidren += 1
            length = len(list(self.conv_layers.children()))
            if idx_chidren < length + 1:
                for param in child.parameters():
                    param.requires_grad = False
        self.loss_criterion = nn.BCELoss(reduction='mean')

        ############################################################################
        # Student code end
        ############################################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass with the net, duplicating grayscale channel to 3-channel.

        Args:
            x: tensor of shape (N,C,H,W) representing input batch of images
        Returns:
            y: tensor of shape (N,num_classes) representing the output (raw scores) of the net
                Note: we set num_classes=15
        """
        model_output = None
        x2 = x.repeat(1, 3, 1, 1)  # as ResNet accepts 3-channel color images
        ############################################################################
        # Student code begin
        ############################################################################
        x2=x2.clone()
        x2 = self.conv_layers(x2).squeeze(2).squeeze(2)
        x2=self.fc_layers(x2)
        model_output = self.activation(x2)

        ############################################################################
        # Student code end
        ############################################################################
        return model_output
