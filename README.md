# TVT-2024-GAN-GRU-Based-Space-Time-Predictive-Channel-Model
This code provides the channel measurement data, source files, and generation codes of all the figure in this paper:
Z. Li, C.-X. Wang*, C. Huang*, J. Huang, J. Li, W. Zhou, and Y. Chen, "A GAN-GRU Based Space-Time Predictive Channel Model for 6G Wireless Communications," IEEE Transactions on Vehicular Technology, vol. 73, no. 7, pp. 9370-9386, July 2024

@article{Li2024GANGRU,
  title={A GAN-GRU Based Space-Time Predictive Channel Model for 6G Wireless Communications},
  author={Li, Zheao and Wang, Cheng-Xiang and Huang, Chen and Huang, Jie and Li, Junling and Zhou, Wenqi and Chen, Yunfei},
  journal={IEEE Transactions on Vehicular Technology},
  year={2024},
  doi={10.1109/tvt.2024.3367386}
}

# Main Modules
The project implements a comprehensive framework for channel modeling, consisting of three main modules:
1. Path Identification: A threshold-based algorithm to classify LoS/NLoS paths from raw measurement data.
2. STGAN (Spatio-Temporal GAN): A Conditional GAN (CGAN) based model for high-fidelity channel data augmentation.
3. GRU Predictor: A Gated Recurrent Unit network for time-series channel prediction.

The repository focuses on the three core algorithms described in the paper:
1. LOS_NLOS.m: MATLAB script for Algorithm 1 (Delay PSD-Based Path Identification). It processes raw Channel Impulse Response (CIR) data and identifies Line-of-Sight (LoS) and Non-Line-of-Sight (NLoS) paths based on physical thresholds.
2. STGAN.py: Python implementation of Algorithm 2 (STGAN-Based Data Augmentation). It trains a conditional generative model to synthesize realistic CIR data.
3. GRU_DPSD.py: Python implementation of Algorithm 2 (Predictive Model). It uses the augmented dataset to train a GRU network for predicting future channel parameters.

# Requirements
To run the codes, you need the following environment:
1. Python Environment (for STGAN.py & GRU_DPSD.py)：Python 3.8+, PyTorch (Torch), NumPy, Pandas, Scikit-learn, Matplotlib
2. MATLAB Environment (for LOS_NLOS.m): MATLAB R2020b or later
