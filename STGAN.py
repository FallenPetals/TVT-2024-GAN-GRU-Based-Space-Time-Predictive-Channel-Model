import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

# Configuration
DATA_FILE_PATH = 'LOS_GAN_origin.xls'
BATCH_SIZE = 4  # 样本较少时减小 Batch Size
LR = 0.0002
N_CRITIC = 5  # 判别器更新频率
NOISE_DIM = 100  # 噪声维度
EPOCHS = 2000  # 增加训练轮数以保证收敛


# STGAN Architecture
class Generator(nn.Module):
    def __init__(self, noise_dim, condition_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim + condition_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    def forward(self, z, condition):
        x = torch.cat([z, condition], dim=1)
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim, condition_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + condition_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, h, condition):
        x = torch.cat([h, condition], dim=1)
        return self.model(x)


# data loding
def load_and_fix_data(file_path):
    print(f"Loading data from {file_path}...")

    try:
        if file_path.endswith('.xls') or file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path, header=None)
        else:
            df = pd.read_csv(file_path, header=None)

        print(f"Raw Shape: {df.shape}")

        # 我们测量的真实信道数据 (300行, 15列)，因此转置为 (15样本, 300特征)
        if df.shape[0] > df.shape[1]:
            print("Transposing...")
            df = df.T

        data_values = df.values.astype(float)
        print(f" (Samples, Features): {data_values.shape}")  # 期望 (15, 300)

        # Generate condition
        conditions = np.mean(data_values, axis=1).reshape(-1, 1)

        # Normalization
        scaler = MinMaxScaler(feature_range=(-1, 1))
        data_normalized = scaler.fit_transform(data_values)

        cond_scaler = MinMaxScaler(feature_range=(0, 1))
        conditions_normalized = cond_scaler.fit_transform(conditions)

        tensor_data = torch.FloatTensor(data_normalized)
        tensor_conditions = torch.FloatTensor(conditions_normalized)

        return tensor_data, tensor_conditions, scaler

    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == '__main__':

    # 1. 加载数据
    real_signals, conditions, scaler = load_and_fix_data(DATA_FILE_PATH)

    if real_signals is not None:
        signal_dim = real_signals.shape[1]
        condition_dim = conditions.shape[1]

        dataset = TensorDataset(real_signals, conditions)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        # 2. 初始化模型
        G = Generator(NOISE_DIM, condition_dim, signal_dim)
        D = Discriminator(signal_dim, condition_dim)

        # 3. 优化器
        optimizer_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
        optimizer_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

        criterion = nn.BCELoss()

        print("Start Training...")

        for epoch in range(EPOCHS):
            for i, (batch_real, batch_cond) in enumerate(dataloader):
                current_batch_size = batch_real.size(0)
                real_labels = torch.ones(current_batch_size, 1)
                fake_labels = torch.zeros(current_batch_size, 1)

                # --- Train D ---
                optimizer_D.zero_grad()
                outputs_real = D(batch_real, batch_cond)
                d_loss_real = criterion(outputs_real, real_labels)

                z = torch.randn(current_batch_size, NOISE_DIM)
                fake_signals = G(z, batch_cond)
                outputs_fake = D(fake_signals.detach(), batch_cond)
                d_loss_fake = criterion(outputs_fake, fake_labels)

                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                optimizer_D.step()

                # --- Train G ---
                if i % N_CRITIC == 0:
                    optimizer_G.zero_grad()
                    z = torch.randn(current_batch_size, NOISE_DIM)
                    gen_signals = G(z, batch_cond)
                    outputs = D(gen_signals, batch_cond)
                    g_loss = criterion(outputs, real_labels)
                    g_loss.backward()
                    optimizer_G.step()

            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{EPOCHS}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

        print("Training Finished.")

        # ==========================================
        # 6. 生成并反归一化 (关键修正)
        # ==========================================
        with torch.no_grad():
            # 模拟生成 1000 条数据
            num_generated = 1000
            test_z = torch.randn(num_generated, NOISE_DIM)

            # 随机从真实条件中采样
            random_indices = np.random.randint(0, len(conditions), num_generated)
            test_conditions = conditions[random_indices]

            # 生成 (此时数据在 -1 到 1 之间)
            gen_data_norm = G(test_z, test_conditions).numpy()

            # 反归一化还原到 -90 ~ -35 dBm
            gen_data_denorm = scaler.inverse_transform(gen_data_norm)

            # 转置回 (300, N) 格式，每一列是一条 CIR
            df_gen = pd.DataFrame(gen_data_denorm).T

            # === 修改点：添加表头 ===
            # 创建类似于 "CIR_Sample_0", "CIR_Sample_1" 的列名
            # 明确表示每一列是一个样本
            column_names = [f"Data_Sample_{i}" for i in range(num_generated)]
            df_gen.columns = column_names

            save_name = 'generated_data.csv'
            df_gen.to_csv(save_name, index=False)
            print(f"Save to '{save_name}'")
            print(f"Shapes of generated data: {df_gen.shape} ")
            print(f"Ranges of generated data: Min={df_gen.min().min():.2f}, Max={df_gen.max().max():.2f}")