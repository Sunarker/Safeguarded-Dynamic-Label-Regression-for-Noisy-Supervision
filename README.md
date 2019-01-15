# Safeguarded-Dynamic-Label-Regression-for-Noisy-Supervision

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


### Step through the codes.
1. Form the noisy datasets.
   ```Shell
   python dataset.py
   ```

2. Train the model
  ```Shell
  python cifar10_train.py --train_dir results/events_ce/cifar10_train  # You can train other models like this one
  ```

3. Test and evaluate
  ```Shell
  python cifar10_eval.py --checkpoint_dir results_ce/cifar10_train --eval_dir results/cifar10_eval 
  ```
  
4. Visualization with Tensorboard
  ```Shell
  tensorboard --logdir=results/events_ce --port=8080
  ```
