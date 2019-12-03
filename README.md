# Many Task Learning With Task Routing - [[ICCV'19](http://iccv2019.thecvf.com/) Oral]

This is the official implementation repo for our 2019 ICCV paper [Many Task Learning With Task Routing](http://openaccess.thecvf.com/content_ICCV_2019/html/Strezoski_Many_Task_Learning_With_Task_Routing_ICCV_2019_paper.html):

**Many Task Learning With Task Routing**  
[Gjorgji Strezoski](https://staff.fnwi.uva.nl/g.strezoski/), [Nanne van Noord](https://nanne.github.io/), [Marcel Worring](https://staff.fnwi.uva.nl/m.worring/)  
International Conference on Computer Vision ([ICCV](http://iccv2019.thecvf.com/)), 2019 [Oral]  
[[CVF](http://openaccess.thecvf.com/content_ICCV_2019/html/Strezoski_Many_Task_Learning_With_Task_Routing_ICCV_2019_paper.html)]  [[ArXiv](https://arxiv.org/abs/1903.12117)] [[Web](https://staff.fnwi.uva.nl/g.strezoski/post/iccv/)]

It contains the Task Routing Layer implentation, its integration in existing models and usage instructions.

---

![Figure 1](https://github.com/gstrezoski/taskrouting/blob/master/fig_2.png)

**Abstract:**  Typical multi-task learning (MTL) methods rely on architectural adjustments and a large trainable parameter set to jointly optimize over several tasks. However, when the number of tasks increases so do the complexity of the architectural adjustments and resource requirements. In this paper, we introduce a method which applies a conditional feature-wise transformation over the convolutional activations that enables a model to successfully perform a large number of tasks. To distinguish from regular MTL, we introduce Many Task Learning (MaTL) as a special case of MTL where more than 20 tasks are performed by a single model. Our method dubbed Task Routing (TR) is encapsulated in a layer we call the Task Routing Layer (TRL), which applied in an MaTL scenario successfully fits hundreds of classification tasks in one model. We evaluate on 5 datasets and the Visual Decathlon (VD) challenge against strong baselines and state-of-the-art approaches. 

---

### Usage

#### Task Routing Layer

In `taskrouting.py` you can find the Task Routing Layer source. It is a standalone file containing the PyTorch layer class. It takes 3 input arguments for instantiation:

- `unit_count  (int)`: Number of input channels going into the Task Routing layer (TRL).
- `task_count  (int)`: Number of tasks. (In Single Task Learning it applies to number of output classes)
- `sigma (float)`: Ratio for routed units per task. (0.5 is default)

### Sample Model

In `routed_vgg.py` you can find an implementation of the stock PyTorch VGG-11 model with or without BatchNorm converted for brahnched MTL. With:

```python
for ix in range(self.task_count):
  self.add_module("classifier_" + str(ix), nn.Sequential(
  nn.Linear(1024 * bottleneck_spatial[0] * bottleneck_spatial[1], 2)
  ))
```

we create as many task specific branches as there are tasks. Additionally, the forward function is designed to forward the data through the active task branch only. 

In the code snippet (lines 71 to 74 from `routed_vgg.py`) below we add the TRL to the VGG net:

```python
conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
router = TaskRouter(v, task_count, int(v * sigma), "taskrouter_"+str(ix))
if batch_norm:
   layers += [conv2d, nn.BatchNorm2d(v), router, nn.ReLU(inplace=True)]
```

For training a model with the Task Routing Layer, the active model task needs to be randomly changed over the training itterations within an epoch. For example:

```python
def change_task(m):
    if hasattr(m, 'active_task'):
        m.set_active_task(active_task)


def train(args, model, task_count, device, train_loader, optimizer, criterion, epoch, total_itts):

    train_start = time.time()
    model.train()

    correct, positives, true_positives, score_list = initialize_evaluation_vars()

    epoch_loss = 0
    individual_loss = [0 for i in range(task_count)]

    for enum_return in enumerate(train_loader):

        batch_idx = enum_return[0]
        data = enum_return[1][0]
        targets = enum_return[1][1:]

        data = data.to(device)
        
        for ix in sample(range(task_count), 1):
            target = targets[ix].to(device)
            global active_task
            active_task = ix

            model = model.apply(change_task)
            out = model(data)
            labels = target[:, ix]
            train_loss = criterion(out, labels)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

    train_end = time.time()
    print("Execution time:", train_end - train_start, "s.")

    return total_itts
```

If you find this repository usefull, please cite this paper:

```
@article{strezoski2019taskrouting,
title={Many Task Learning With Task Routing},
author={Strezoski, Gjorgji and van Noord, Nanne and Worring, Marcel},
booktitle = {International Conference on Computer Vision (ICCV)},
organization={IEEE}
year={2019},
url={https://arxiv.org/abs/1903.12117}
}
```
