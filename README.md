# TVT-2024-GAN-GRU-Based-Space-Time-Predictive-Channel-Model
This code provides the channel measurement data, source files, and generation codes of all the figure in this paper:
Z. Li, C.-X. Wang*, C. Huang*, J. Huang, J. Li, W. Zhou, and Y. Chen, "A GAN-GRU Based Space-Time Predictive Channel Model for 6G Wireless Communications," IEEE Transactions on Vehicular Technology, vol. 73, no. 7, pp. 9370-9386, July 2024

# Structure
The project implements a comprehensive framework for channel modeling, consisting of three main modules:
1. Path Identification: A threshold-based algorithm to classify LoS/NLoS paths from raw measurement data.
2. STGAN (Spatio-Temporal GAN): A Conditional GAN (CGAN) based model for high-fidelity channel data augmentation.
3. GRU Predictor: A Gated Recurrent Unit network for time-series channel prediction.
