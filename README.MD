# Tutorial Adversarial Machine Learning
## Focusing on Adversarial Examples

This document supports a tutorial on adversarial examples presented at [WIFS 2019](https://wifs2019.tudelft.nl/?q=node/12) and it is an interactive extension of a [larger survey](https://arxiv.org/abs/1810.01185) made publicly available before the tutorial.
Since the event is focused on security and expects participants with various backgrounds in the field, the first part of the tutorial is dedicated to an introduction to machine learning, describing the general setting and the most important assumptions. In order to maintain focus on security issues, we use examples from supervised classification - a branch of machine learning - although the information contained in this notebook can easily be extended to regression, unsupervised classification or reinforcement learning.
Whenever possible. pointers and references to these fields are given.

The second part presents a taxonomy on security of machine learning and gives references to relevant works for each attack type.

The third  (and last) part of the tutorial dives into adversarial inputs, a type of attack at test (inference) time.

Transcript notes will soon be added.


# Run
The tutorial uses jupyter notebook [slides](https://github.com/damianavila/RISE).
It assumes you already have installed python and the jupyter notebooks environment.
In order to install all other dependencies, clone the project and run:

```
$ pip install -r requirements.txt
```


Afterwards, run ```jupyter notebook``` and select the notebook.

Using the first notebook cell you can hide or reveal the implementation details.

In order to run a cell you can use ```shift+enter```. 
While in presentation mode, you can use ```space``` to go forwarard and ```shift+space``` to go backwards.
