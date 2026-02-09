# TVT-2024-GAN-GRU-Based-Space-Time-Predictive-Channel-Model
This code provides the channel measurement data, source files, and generation codes of all the figure in this paper:
Z. Li, C.-X. Wang*, C. Huang*, J. Huang, J. Li, W. Zhou, and Y. Chen, "A GAN-GRU Based Space-Time Predictive Channel Model for 6G Wireless Communications," IEEE Transactions on Vehicular Technology, vol. 73, no. 7, pp. 9370-9386, July 2024.

If you use this data or code, please cite the following paper:

@article{Li2024GANGRU,
  title={A GAN-GRU Based Space-Time Predictive Channel Model for 6G Wireless Communications},
  author={Li, Zheao and Wang, Cheng-Xiang and Huang, Chen and Huang, Jie and Li, Junling and Zhou, Wenqi and Chen, Yunfei},
  journal={IEEE Transactions on Vehicular Technology},
  year={2024},
  doi={10.1109/tvt.2024.3367386}
}


# Channel Measurement Data
The raw data is organized into .mat files, categorized by measurement scenarios and data types.

Indoor Multi-Band Measurement Data (2.4 / 5 / 6 GHz): This subset includes channel impulse responses (CIRs) and power delay profiles (PDPs) collected in an indoor corridor environment. It covers three key frequency bands, providing a rich basis for frequency-dependent channel analysis.

•	APDP_LOS_24G_new2.mat: Average Power Delay Profile (APDP) under Line-of-Sight (LoS) conditions at 2.4 GHz Band.

•	APDP_NLOS_24G_new2.mat: APDP under Non-Line-of-Sight (NLoS) conditions at 2.4 GHz Band.

•	APDP_LOS_5G_new2.mat / APDP_NLOS_5G_new2.mat: APDP data for LoS/NLoS scenarios at 5 GHz Band.

•	APDP_LOS_6G_new2.mat / APDP_NLOS_6G_new2.mat: APDP data for LoS/NLoS scenarios at 6 GHz Band.

Outdoor UMi Scenario Data: This subset represents an Urban Micro-cell (UMi) environment, featuring continuous route measurements to capture spatial channel evolution.

•	CIR_L_case1_15pos.mat: Channel measurement data collected at 15 sequential positions collected in the LoS scenario.

•	CIR_N_case1_10pos.mat: Channel measurement data collected at 10 sequential positions collected in the NLoS scenario.


# Main Modules
The project implements a comprehensive framework for channel modeling, consisting of three main modules:
1. Path Identification: A threshold-based algorithm to classify LoS/NLoS paths from raw measurement data.
2. STGAN (Space-Time GAN): A Conditional GAN (CGAN) based model for high-fidelity channel data augmentation.
3. GRU Predictor: A Gated Recurrent Unit network for time-series channel prediction.

The repository focuses on the three core algorithms described in the paper:
1. LOS_NLOS.m: MATLAB script for Algorithm 1 (Delay PSD-Based Path Identification). It processes raw Channel Impulse Response (CIR) data and identifies Line-of-Sight (LoS) and Non-Line-of-Sight (NLoS) paths based on physical thresholds.
2. STGAN.py: Python implementation of Algorithm 2 (STGAN-Based Data Augmentation). It trains a conditional generative model to synthesize realistic CIR data.
3. GRU_DPSD.py: Python implementation of Algorithm 2 (Predictive Model). It uses the augmented dataset to train a GRU network for predicting future channel parameters.


# Requirements
To run the codes, you need the following environment:
1. Python Environment (for STGAN.py & GRU_DPSD.py)：Python 3.8+, PyTorch (Torch), NumPy, Pandas, Scikit-learn, Matplotlib
2. MATLAB Environment (for LOS_NLOS.m): MATLAB R2020b or later
