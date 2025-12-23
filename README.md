# EXTRA 
This is the official implementation of our NeurIPS 2025 paper "Exploring Tradeoffs through Mode Connectivity for Multi-Task Learning".

## Image-to-Image Prediction
The supervised multitask learning experiments are conducted on NYU-v2 and CityScapes datasets. We follow the setup from [MTAN](https://github.com/lorenmt/mtan). The datasets could be downloaded from [NYU-v2](https://www.dropbox.com/sh/86nssgwm6hm3vkb/AACrnUQ4GxpdrBbLjb6n-mWNa?dl=0) and [CityScapes](https://www.dropbox.com/sh/gaw6vh6qusoyms6/AADwWi0Tp3E3M4B2xzeGlsEna?dl=0). After the datasets are downloaded, please follow the respective run.sh script in each folder. In particular, modify the dataroot variable to the downloaded dataset.

### Run Experiment
#### Training (Two-Step):
First Step: Train your own endpoints by LS (or other MTL approaches).

Second Step: Specify the `init_start` and `init_end` with your trained weights in the first step, then start training curve.

The dataset by default should be put under `experiments/EXP_NAME/dataset/` folder where `EXP_NAME` is chosen from `{celeba, cityscapes, nyuv2, quantum_chemistry}`. To run the experiment:
```
cd cityscapes/
sh run.sh
```

#### Evaluation
Specify the `ckpt` in evaluate_path.py, then:
```
python evaluate_path.py
```

## Correspondence
If you have any further questions, please feel free to contact Zhipeng Zhou by zzp1994@mail.ustc.edu.cn

## Acknowledgements
We would like to acknowledge the contribution of [FairGrad](https://github.com/OptMN-Lab/fairgrad) to the base codes.

## Citations
If you find our work interesting or the repo useful, please consider citing this paper:
```
@article{zhouexploring,
  title={Exploring Tradeoffs through Mode Connectivity for Multi-Task Learning},
  author={Zhou, Zhipeng and Meng, Ziqiao and Wu, Pengcheng and Zhao, Peilin and Miao, Chunyan},
  journal={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025}
}
```
