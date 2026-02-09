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


# Channel Measurement Data
The raw data is organized into .mat files, categorized by measurement scenarios and data types.

1. Indoor Multi-Band Measurement Data (2.4 / 5 / 6 GHz): This subset includes channel impulse responses (CIRs) and power delay profiles (PDPs) collected in an indoor corridor environment. It covers three key frequency bands, providing a rich basis for frequency-dependent channel analysis.

   1. APDP_LOS_24G_new2.mat: Average power delay profile (APDP) under Line-of-Sight (LoS) conditions at 2.4 GHz Band.
   2. APDP_NLOS_24G_new2.mat: APDP under Non-Line-of-Sight (NLoS) conditions at 2.4 GHz Band.
   3. APDP_LOS_5G_new2.mat / APDP_NLOS_5G_new2.mat: APDP data for LoS/NLoS scenarios at 5 GHz Band.
   4. APDP_LOS_6G_new2.mat / APDP_NLOS_6G_new2.mat: APDP data for LoS/NLoS scenarios at 6 GHz Band.

2. Outdoor UMi Scenario Data: This subset represents an Urban Micro-cell (UMi) environment, featuring continuous route measurements to capture spatial channel evolution.
   1. CIR_L_case1_15pos.mat: Channel measurement data collected at 15 sequential positions collected in the LoS scenario.
   2. CIR_N_case1_10pos.mat: Channel measurement data collected at 10 sequential positions collected in the NLoS scenario.


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
1. Preprocessing: Run LOS_NLOS.m in MATLAB to process raw channel measurement data and obtain path labels.
2. Data augmentation: Run STGAN.py to train the GAN model and generate sufficient synthetic channel data.
3. Prediction: Run GRU_DPSD.py using the generated data to evaluate prediction performance.
