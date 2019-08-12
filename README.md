# Many Task Learning With Task Routing - [[ICCV'19](http://iccv2019.thecvf.com/) Oral]

This is the official implementation repo for our 2019 ICCV paper [Many Task Learning With Task Routing](https://arxiv.org/abs/1903.12117):

**Many Task Learning With Task Routing**  
[Gjorgji Strezoski](https://staff.fnwi.uva.nl/g.strezoski/), [Nanne van Noord](https://nanne.github.io/), [Marcel Worring](https://staff.fnwi.uva.nl/m.worring/)  
International Conference on Computer Vision ([ICCV](http://iccv2019.thecvf.com/)), 2019 [Oral]  
[[ArXiv](https://arxiv.org/abs/1903.12117)] [[Web](https://staff.fnwi.uva.nl/g.strezoski/post/iccv/)]

It contains the Task Routing Layer implentation, its integration in existing models and usage instructions.

---

![Figure 1](https://github.com/gstrezoski/taskrouting/blob/master/fig_2.png)

**Abstract:**  Typical multi-task learning (MTL) methods rely on architectural adjustments and a large trainable parameter set to jointly optimize over several tasks. However, when the number of tasks increases so do the complexity of the architectural adjustments and resource requirements. In this paper, we introduce a method which applies a conditional feature-wise transformation over the convolutional activations that enables a model to successfully perform a large number of tasks. To distinguish from regular MTL, we introduce Many Task Learning (MaTL) as a special case of MTL where more than 20 tasks are performed by a single model. Our method dubbed Task Routing (TR) is encapsulated in a layer we call the Task Routing Layer (TRL), which applied in an MaTL scenario successfully fits hundreds of classification tasks in one model. We evaluate on 5 datasets and the Visual Decathlon (VD) challenge against strong baselines and state-of-the-art approaches. 

---

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
