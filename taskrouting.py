import torch.nn as nn
import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TaskRouter(nn.Module):

    r""" Applies task specific masking out individual units in a layer.

        Args:
        unit_count  (int): Number of input channels going into the Task Routing layer.
        task_count  (int): Number of tasks. (IN STL it applies to number of output classes)
        sigma (int): Ratio for routed units per task.
    """

    def __init__(self, unit_count, task_count, sigma, name="TaskRouter"):

        super(TaskRouter, self).__init__()

        self.use_routing = True
        self.name = name
        self.unit_count = unit_count
        # Just initilize it with 0. This gets changed right after the model is loaded so the value is never used.
        # We store the active mask for the Task Routing Layer here.
        self.active_task = 0

        if sigma!=0:
            self._unit_mapping = torch.ones((task_count, unit_count))
            self._unit_mapping[np.arange(task_count)[:, None], np.random.rand(task_count, unit_count).argsort(1)[:, :sigma]] = 0
            self._unit_mapping = torch.nn.Parameter(self._unit_mapping)
        else:
            self._unit_mapping = torch.ones((task_count, unit_count))
            self.use_knockout = False
            print("Not using Routing! Sigma is 0")

    def get_unit_mapping(self):

        return self._unit_mapping

    def set_active_task(self, active_task):

        self.active_task = active_task
        return active_task

    def forward(self, input):

        if not self.use_routing:
            return input

        mask = torch.index_select(self._unit_mapping, 0, (torch.ones(input.shape[0])*self.active_task).long().to(device))\
            .unsqueeze(2).unsqueeze(3)
        input.data.mul_(mask)

        return input
