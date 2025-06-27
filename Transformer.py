import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# -----------------------------
# 하이퍼파라미터 설정
# -----------------------------
TARGET_COLS = ['1_1_water', '1_2_water', '1_3_water', '2_1_water', '2_2_water', '2_3_water']
FEATURE_COLS = [
    '1_1_temp', '1_2_temp', '1_3_temp', '2_1_temp', '2_2_temp', '2_3_temp',
    '1_1_ph', '1_2_ph', '1_3_ph', '2_1_ph', '2_2_ph', '2_3_ph',
    '1_1_ec', '1_2_ec', '1_3_ec', '2_1_ec', '2_2_ec', '2_3_ec',
    'precip', 'humidity', 'sun_rad']
HORIZONS = [30, 90, 180, 360]
WINDOW_MAP = {30: 90, 90: 180, 180: 360, 360: 540}
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3
EMBED_DIM = 128
NHEAD = 4
NUM_LAYERS = 2
RES_PATH = "/media/user/AI_2T/UML_Paper/Daytime/Transformer"

# -----------------------------
# Dataset
# -----------------------------
class SequenceDataset(Dataset):
    def __init__(self, X, y, window, horizon):
        self.X = X
        self.y = y
        self.window = window
        self.horizon = horizon

    def __len__(self):
        return len(self.X) - self.window - self.horizon

    def __getitem__(self, idx):
        x = self.X[idx:idx + self.window]
        y = self.y[idx + self.window + self.horizon - 1]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# -----------------------------
# Transformer 기반 모델
# -----------------------------
class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, output_dim):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        out = self.transformer(x)
        return self.fc(out[:, -1, :])

# -----------------------------
# 메인 함수
# -----------------------------
def main():
    df = pd.read_csv(f"{RES_PATH}/../../Main_Data_Jeju/JeJu_merged.csv", parse_dates=['date'])
    df = df.dropna().reset_index(drop=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results, rows, all_preds = {}, [], []
    for horizon in HORIZONS:
        window = WINDOW_MAP[horizon]
        for col in TARGET_COLS:
            X = df[FEATURE_COLS].values
            y = df[[col]].values
            x_scaler = StandardScaler()
            y_scaler = StandardScaler()
            X = x_scaler.fit_transform(X)
            y = y_scaler.fit_transform(y)

            dataset = SequenceDataset(X, y, window, horizon)
            train_size = int(len(dataset)*0.8)
            train_set, val_set = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])
            train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_set, BATCH_SIZE)

            model = SimpleTransformer(
                input_dim=X.shape[1],
                embed_dim=EMBED_DIM,
                num_heads=NHEAD,
                num_layers=NUM_LAYERS,
                output_dim=1
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            loss_fn = nn.MSELoss()

            model.train()
            for epoch in range(EPOCHS):
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb).squeeze()
                    loss = loss_fn(pred, yb.squeeze())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            model.eval()
            y_true_list, y_pred_list = [], []
            dates = df['date'].values[window + horizon - 1:][:len(dataset)]
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    out = model(xb).squeeze()
                    y_true_list.append(yb.numpy())
                    y_pred_list.append(out.cpu().numpy())

            y_true = y_scaler.inverse_transform(np.concatenate(y_true_list).reshape(-1, 1)).flatten()
            y_pred = y_scaler.inverse_transform(np.concatenate(y_pred_list).reshape(-1, 1)).flatten()

            for d, t, p in zip(dates[:len(y_true)], y_true, y_pred):
                all_preds.append([col, f"t+{horizon}", d, t, p])

            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            results[(col, f"t+{horizon}")] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}
            rows.append([col, f"t+{horizon}", mae, mse, rmse, r2])

    df_result = pd.DataFrame(rows, columns=["Well", "Horizon", "MAE", "MSE", "RMSE", "R2"])
    df_result.to_csv(f"{RES_PATH}/transformer_prediction_summary_nolag.csv", index=False)
    pd.DataFrame(all_preds, columns=["Well", "Horizon", "Date", "True", "Pred"]).to_csv(f"{RES_PATH}/transformer_predictions_nolag.csv", index=False)

    os.makedirs(f"{RES_PATH}/plots", exist_ok=True)
    with PdfPages(f"{RES_PATH}/plots/transformer_report_nolag.pdf") as pdf:
        fig, ax = plt.subplots(figsize=(12, 0.5 + len(df_result)*0.25))
        ax.axis('off')
        tbl = ax.table(cellText=df_result.round(4).values, colLabels=df_result.columns,
                       loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.0, 1.5)
        pdf.savefig(); plt.close()

        for m in ['MAE', 'MSE', 'RMSE', 'R2']:
            pivot = df_result.pivot(index='Well', columns='Horizon', values=m).astype(float)
            plt.figure(figsize=(10, 6))
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu")
            plt.title(f"{m} Heatmap (Transformer No LAG)")
            plt.xlabel("Horizon")
            plt.ylabel("Well")
            plt.tight_layout()
            pdf.savefig(); plt.close()

        pred_df = pd.read_csv(f"{RES_PATH}/transformer_predictions_nolag.csv")
        layout_pairs = [(0, 1), (2, 3)]
        for well in TARGET_COLS:
            for pair in layout_pairs:
                fig, axs = plt.subplots(2, 1, figsize=(10, 8))
                for j, h_idx in enumerate(pair):
                    h = HORIZONS[h_idx]
                    sub = pred_df[(pred_df['Well'] == well) & (pred_df['Horizon'] == f"t+{h}")]
                    axs[j].plot(sub["Date"], sub["True"], label="True", color='blue', linewidth=2.0)
                    axs[j].plot(sub["Date"], sub["Pred"], label="Predicted", color='red', linewidth=1.2)
                    axs[j].set_title(f"{well} - t+{h}")
                    axs[j].set_xlabel("Date")
                    axs[j].set_ylabel("Water Level")
                    axs[j].tick_params(axis='x', rotation=45)
                    axs[j].xaxis.set_major_locator(plt.MaxNLocator(6))
                    axs[j].legend()
                plt.suptitle(f"{well} Forecast (t+{HORIZONS[pair[0]]} / t+{HORIZONS[pair[1]]})", fontsize=14)
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                pdf.savefig(); plt.close()

if __name__ == '__main__':
    main()
