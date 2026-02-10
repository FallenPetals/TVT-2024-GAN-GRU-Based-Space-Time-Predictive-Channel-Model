# TVT-2024-GAN-GRU-Based-Space-Time-Predictive-Channel-Model
This repository contains the source code, channel measurement data, and generation scripts for the paper:
Z. Li, C.-X. Wang*, C. Huang*, J. Huang, J. Li, W. Zhou, and Y. Chen, "A GAN-GRU Based Space-Time Predictive Channel Model for 6G Wireless Communications," IEEE Transactions on Vehicular Technology, vol. 73, no. 7, pp. 9370-9386, July 2024.


# Citation
If you use this data or code in your research, please cite the following paper:

@article{Li2024GANGRU,
  title={A GAN-GRU Based Space-Time Predictive Channel Model for 6G Wireless Communications},
  author={Li, Zheao and Wang, Cheng-Xiang and Huang, Chen and Huang, Jie and Li, Junling and Zhou, Wenqi and Chen, Yunfei},
  journal={IEEE Transactions on Vehicular Technology},
  year={2024},
  volume={73},
  number={7},
  pages={9370--9386},
  doi={10.1109/tvt.2024.3367386}
}


# Main Modules
The project implements a comprehensive framework for channel modeling, consisting of three main modules:
1. Path Identification (LOS_NLOS.m)
   1. Description: MATLAB script for Algorithm 1 (Delay PSD-Based Path Identification).
   2. Function: It processes raw channel measurement CIR data and identifies LoS/NLoS paths.
   3. Key Logic: The algorithm strictly utilizes physical thresholds ($S_{max} \ge T_1$ and $S_{mean} \ge T_2$) to classify paths, ensuring robust identification performance.

2. STGAN Data Augmentation (STGAN.py)
   1. Description: Python implementation of Algorithm 2 (STGAN-Based Data Augmentation).
   2. Function: Trains a Conditional GAN (CGAN) to synthesize realistic CIR data.
   3. Key Logic: Implements a standard CGAN architecture where the generator explicitly concatenates noise ($z$) and condition ($y$) as input (G(z|y)), and uses Binary Cross Entropy (Log Loss)  for stable training.

3. Channel Prediction (GRU_DPSD.py)
      1. Description: Python implementation of the predictive model.
      2. Function: Utilizes the STGAN-augmented dataset to train a Gated Recurrent Unit (GRU) network.
      3. Key Logic: Performs time-series prediction of future channel parameters based on historical channel evolution.


# Requirements
To run the codes, you need the following environment:
1. Python Environment (for STGAN.py & GRU_DPSD.py)：Python 3.8+, PyTorch (Torch), NumPy, Pandas, Scikit-learn, Matplotlib, scipy.
2. MATLAB Environment (for LOS_NLOS.m): MATLAB R2020b or later.


# Usage
1. Preprocessing: Run LOS_NLOS.m in MATLAB to process raw channel measurement data and obtain path labels. Manually convert the preprocessed data from .mat file to .csv file for Step 2.
2. Data augmentation: Run STGAN.py to train the GAN model and generate sufficient synthetic channel data.
3. Prediction: Run GRU_DPSD.py using the generated data to evaluate prediction performance.


# Notes
This research constitutes a core component of the project "Research on Machine Learning-based Predictive Channel Modeling", collaborated with Huawei Technologies Co., Ltd., which has been successfully completed and validated by the industry partner.

Please note that due to commercial confidentiality and data privacy policies associated with this industrial project, the full proprietary channel measurement database and specific internal system integration codes cannot be publicly released. 

However, to ensure the reproducibility and scientific integrity of this paper, we have explicitly open-sourced the validated core algorithms (including Algorithm 1 & 2) and the corresponding measurement datasets used to generate the reported results.
