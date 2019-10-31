import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.linux_wrapper import create_dir


class MLPModel(nn.Module):
    """Simple multi-layer perception model

    Attributes:
        config(Config cls): configurations of params
        image_height(int): input image height
        image_width(int): input image width
        hidden_size(list(int)): number of neuron units in each layer
        layers(list(nn.Linear)): layers of model

    """
    def __init__(self, config):
        super(MLPModel, self).__init__()
        self.config = config
        self.image_height = config.image_height
        self.image_width = config.image_width
        self.hidden_size = config.hidden_size
        self.h1 = nn.Linear(self.image_height * self.image_width, self.hidden_size[0])
        self.output = nn.Linear(self.hidden_size[0], self.hidden_size[1])

    def forward(self, src):
        """Forward function
        Args:
            src(list(array)): list of images, shape: [(H, W), ... ]

        Returns:
            out(tensor): output of input images, tensor shape:(batch_size, output_size)
        """
        batch_size = len(src)
        height = src[0].shape[0]
        width = src[0].shape[1]
        _src = torch.tensor(src, device=self.device).view(batch_size, height * width).float()

        _src = F.relu(self.h1(_src))
        out = self.output(_src)

        return out

    def save(self, dir, filename):
        """ Save the model to a file

        Args:
            path(str): path to the file
        """
        create_dir(dir)
        print('save model parameters to [%s]' % (dir + filename), file=sys.stdout)

        params = {
            'args': dict(config_model=self.config),
            'state_dict': self.state_dict()
        }

        torch.save(params, dir + filename)

    @classmethod
    def load(cls, model_path):
        """ Load the model from a file

        Args:
            model_path(str): path to the model persistent file

        Returns:
            model(nn.Module): model instance
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = MLPModel(args['config_model'])
        model.load_state_dict(params['state_dict'])

        return model

    @property
    def device(self):
        """Determine which device to place the tensors upon, CPU or GPU

        Returns:
            device(Device): torch.device
        """
        device = self.h1.weight.device
        return device


