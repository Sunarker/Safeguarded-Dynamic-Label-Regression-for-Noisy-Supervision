# Safeguarded-Dynamic-Label-Regression-for-Noisy-Supervision  

[[Paper]](https://sunarker.github.io/temp/AAAI2019_Dynamic_Label_Regression_for_Noisy_Supervision.pdf) [[Presentation]](
        https://sunarker.github.io/temp/AAAI2019_Presentation.pdf) [[Poster]](https://sunarker.github.io/temp/AAAI2019_Poster.pdf) 

If you use this code in your research, please cite
```
@inproceedings{aaai19-jiangchao,
  title     = {Safeguarded Dynamic Label Regression for Noisy Supervision},
  author    = {Jiangchao Yao, Hao Wu, Ya Zhang, Ivor W. Tsang and Jun Sun},
  booktitle = {Proceedings of the Association for the Advancement of Artificial Intelligence Conference on
               Artificial Intelligence, {AAAI-19}},
  publisher = {Association for the Advancement of Artificial Intelligence Conference},
  year      = {2019}
}
```


### Form the noisy datasets.
  ```Shell
  # open a new shell
  python dataset.py
  ```

### Train the models.
1. Train DNNs directly with the cross-entropy loss.
  ```Shell
  # open a new shell
  python cifar10_train.py --train_dir results/events_ce/cifar10_train --noise_rate 0.3 # You can train other models like this one
  ```

2. Train Bootstrapping
```Shell
  # open a new shell
  python cifar10_train_bootstrapping.py --train_dir results/events_bootstrapping/cifar10_train --noise_rate 0.3 # You can train other models like this one
  ```

3. Train Forward 
```Shell
  # open a new shell
  python cifar10_train_T.py --init_dir results/events_ce/cifar10_train --train_dir results/events_T/cifar10_train --noise_rate 0.3 # You can train other models like this one
  ```
  
4. Train S-adaptation
```Shell
  # open a new shell
  python cifar10_train_varT.py --init_dir results/events_ce/cifar10_train --train_dir results/events_varT/cifar10_train --noise_rate 0.3 # You can train other models like this one
  ```
  
5. Train LCCN
```Shell
  # open a new shell
  python cifar10_train_varC.py --init_dir results/events_ce/cifar10_train --train_dir results/events_varC/cifar10_train --noise_rate 0.3 # You can train other models like this one
  ```

### Test and evaluate
  ```Shell
  # open a new shell
  python cifar10_eval.py --checkpoint_dir results/events_ce/cifar10_train --eval_dir results/cifar10_eval 
  ```
  
### Visualization with Tensorboard
  ```Shell
  tensorboard --logdir=results/events_ce --port=8080
  ```

### More running settings can be found by reading arguments in the code.
