# Safeguarded-Dynamic-Label-Regression-for-Noisy-Supervision

If you use this code in your research, please cite
```
@inproceedings{ijcai2018-135,
  title     = {Safeguarded Dynamic Label Regression for Noisy Supervision},
  author    = {Jiangchao Yao, Hao Wu, Ya Zhang, Ivor W. Tsang and Jun Sun},
  booktitle = {Proceedings of the Association for the Advancement of Artificial Intelligence Conference on
               Artificial Intelligence, {AAAI-19}},
  publisher = {Association for the Advancement of Artificial Intelligence Conference},
  year      = {2019}
}
```


### Step through the code.
1. Download pre-trained models and weights. For the pretrained [wsddn](https://www.robots.ox.ac.uk/~vgg/publications/2016/Bilen16/bilen16.pdf) model, you can find the download link [here](https://goo.gl/j7tp7N). For other pre-trained models like VGG16 and Resnet V1 models, they are provided by [pytorch-vgg](https://github.com/jcjohnson/pytorch-vgg.git) and [pytorch-resnet](https://github.com/ruotianluo/pytorch-resnet) (the ones with caffe in the name). You can download them in the ``data/imagenet_weights`` folder. For example for VGG16 model, you can set up like:
   ```Shell
   mkdir -p data/imagenet_weights
   cd data/imagenet_weights
   python # open python in terminal and run the following Python code
   ```
   ```Python
   import torch
   from torch.utils.model_zoo import load_url
   from torchvision import models

   sd = load_url("https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg16-00b39a1b.pth")
   sd['classifier.0.weight'] = sd['classifier.1.weight']
   sd['classifier.0.bias'] = sd['classifier.1.bias']
   del sd['classifier.1.weight']
   del sd['classifier.1.bias']

   sd['classifier.3.weight'] = sd['classifier.4.weight']
   sd['classifier.3.bias'] = sd['classifier.4.bias']
   del sd['classifier.4.weight']
   del sd['classifier.4.bias']

   torch.save(sd, "vgg16.pth")
   ```
   ```Shell
   cd ../..
   ```
   For Resnet101, you can set up like:
   ```Shell
   mkdir -p data/imagenet_weights
   cd data/imagenet_weights
   # download from my gdrive (link in pytorch-resnet)
   mv resnet101-caffe.pth res101.pth
   cd ../..
   ```

2. Train (and test, evaluation)
  ```Shell
  ./experiments/scripts/train.sh [GPU_ID] [DATASET] [NET] [WSDDN_PRETRAINED]
  # Examples:
  ./experiments/scripts/train.sh 0 pascal_voc vgg16 path_to_wsddn_pretrained_model
  ```

3. Visualization with Tensorboard
  ```Shell
  tensorboard --logdir=tensorboard/vgg16/voc_2007_trainval/ --port=7001 &
  ```

4. Test and evaluate
  ```Shell
  ./experiments/scripts/test.sh [GPU_ID] [DATASET] [NET] [WSDDN_PRETRAINED]
  # Examples:
  ./experiments/scripts/test.sh 0 pascal_voc vgg16 path_to_wsddn_pretrained_model
  ```

By default, trained networks are saved under:

```
output/[NET]/[DATASET]/default/
```

Test outputs are saved under:

```
output/[NET]/[DATASET]/default/[SNAPSHOT]/
```

Tensorboard information for train and validation is saved under:

```
tensorboard/[NET]/[DATASET]/default/
tensorboard/[NET]/[DATASET]/default_val/
```
