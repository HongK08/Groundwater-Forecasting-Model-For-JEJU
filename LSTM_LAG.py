import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import os

# ====== 설정 ======
data_path = '/media/user/AI_2T/UML_Paper/Main_Data_Jeju/JeJu_merged.csv'
res_path = '/media/user/AI_2T/UML_Paper/Daytime/LSTM_LAG'
os.makedirs(f'{res_path}/plots', exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# ====== 하이퍼파라미터 ======
BATCH_SIZE = 64
EPOCHS = 100
LR = 0.001
HIDDEN_DIM = 128
NUM_LAYERS = 2

# ====== 대상 설정 ======
TARGET_COLS = ['1_1_water', '1_2_water', '1_3_water', '2_1_water', '2_2_water', '2_3_water']
HORIZONS = [30, 90, 180, 360]
WINDOW_MAP = {30: 90, 90: 180, 180: 360, 360: 540}

# ====== 시각화 설정 ======
TRUE_COLOR = 'blue'
PRED_COLOR = 'red'
TRUE_WIDTH = 2.0
PRED_WIDTH = 1.2

# ====== Dataset 정의 ======
class SequenceDataset(Dataset):
    def __init__(self, X, y, window_size, horizon):
        self.X = X
        self.y = y
        self.window = window_size
        self.horizon = horizon

    def __len__(self):
        return len(self.X) - self.window - self.horizon + 1

    def __getitem__(self, idx):
        X_seq = self.X[idx : idx + self.window]
        y_target = self.y[idx + self.window + self.horizon - 1]
        return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y_target, dtype=torch.float32)

# ====== BiLSTM 모델 정의 ======
class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, NUM_LAYERS, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def run_lstm_pipeline():
    df = pd.read_csv(data_path, parse_dates=['date'])
    df = df.sort_values("date").reset_index(drop=True).ffill()

    # LAG 피처 추가
    for col in TARGET_COLS:
        for lag in [1, 2, 3]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    df = df.dropna().reset_index(drop=True)

    INPUT_COLS = [col for col in df.columns if col not in ["date"] + TARGET_COLS]

    all_preds, metric_table = [], []
    metric_dict = {m: pd.DataFrame(index=TARGET_COLS, columns=[f"t+{h}" for h in HORIZONS]) for m in ['MAE', 'MSE', 'RMSE', 'R2']}

    total_jobs = len(TARGET_COLS) * len(HORIZONS)
    job_counter = 0

    for target_col in TARGET_COLS:
        for horizon in HORIZONS:
            job_counter += 1
            print(f"\n▶ ({job_counter}/{total_jobs}) Target: {target_col}, Horizon: t+{horizon} (Window: {WINDOW_MAP[horizon]})")

            x_scaler = StandardScaler()
            y_scaler = StandardScaler()
            X_scaled = x_scaler.fit_transform(df[INPUT_COLS])
            y_scaled = y_scaler.fit_transform(df[[target_col]])

            window_size = WINDOW_MAP[horizon]
            dataset = SequenceDataset(X_scaled, y_scaled, window_size, horizon)
            train_size = int(len(dataset) * 0.8)
            train_set, val_set = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])
            train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_set, BATCH_SIZE, shuffle=False)

            model = BiLSTMModel(X_scaled.shape[1], HIDDEN_DIM, 1).to(device)
            criterion = nn.SmoothL1Loss()
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)

            for epoch in range(EPOCHS):
                model.train()
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    optimizer.zero_grad()
                    pred = model(X_batch).squeeze(1)
                    target = y_batch.squeeze()
                    loss = criterion(pred, target)
                    loss.backward()
                    optimizer.step()
                if (epoch + 1) % 10 == 0:
                    print(f"    [Epoch {epoch+1}/{EPOCHS}] Loss: {loss.item():.4f}")

            model.eval()
            y_true_list, y_pred_list = [], []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    preds = model(X_batch)
                    y_true_list.append(y_batch)
                    y_pred_list.append(preds.squeeze())

            y_true = torch.cat(y_true_list).cpu().numpy()
            y_pred = torch.cat(y_pred_list).cpu().numpy()
            y_true_inv = y_scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
            y_pred_inv = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            dates = df["date"].values[window_size + horizon - 1:][-len(y_true_inv):]

            df_temp = pd.DataFrame({"Well": target_col, "Horizon": f"t+{horizon}", "Date": dates,
                                    "True": y_true_inv, "Pred": y_pred_inv})
            all_preds.append(df_temp)

            mae = mean_absolute_error(y_true_inv, y_pred_inv)
            mse = mean_squared_error(y_true_inv, y_pred_inv)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true_inv, y_pred_inv)
            print(f"    ✔ Done - MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")

            metric_table.append([target_col, f"t+{horizon}", mae, mse, rmse, r2])
            metric_dict['MAE'].loc[target_col, f"t+{horizon}"] = mae
            metric_dict['MSE'].loc[target_col, f"t+{horizon}"] = mse
            metric_dict['RMSE'].loc[target_col, f"t+{horizon}"] = rmse
            metric_dict['R2'].loc[target_col, f"t+{horizon}"] = r2

    pred_df = pd.concat(all_preds, ignore_index=True)
    pred_df.to_csv(f'{res_path}/bilstm_predictions_with_lag.csv', index=False)

    summary_df = pd.DataFrame(metric_table, columns=["Well", "Horizon", "MAE", "MSE", "RMSE", "R2"])
    summary_df.to_csv(f'{res_path}/bilstm_prediction_summary_with_lag.csv', index=False)

    pdf_path = f'{res_path}/plots/bilstm_with_lag_report.pdf'
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(12, 0.5 + len(summary_df)*0.25))
        ax.axis('off')
        tbl = ax.table(cellText=summary_df.round(4).values, colLabels=summary_df.columns,
                       loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.0, 1.5)
        pdf.savefig()
        plt.close()

        for m in ['MAE', 'MSE', 'RMSE', 'R2']:
            plt.figure(figsize=(10, 6))
            sns.heatmap(metric_dict[m].astype(float), annot=True, fmt=".3f", cmap="YlGnBu")
            plt.title(f"{m} Heatmap (BiLSTM + LAG)")
            plt.xlabel("Horizon")
            plt.ylabel("Well")
            plt.tight_layout()
            pdf.savefig()
            plt.close()

        layout_pairs = [(0, 1), (2, 3)]
        for well in TARGET_COLS:
            for pair in layout_pairs:
                fig, axs = plt.subplots(2, 1, figsize=(10, 8))
                for j, h_idx in enumerate(pair):
                    h = HORIZONS[h_idx]
                    sub = pred_df[(pred_df['Well'] == well) & (pred_df['Horizon'] == f"t+{h}")]
                    axs[j].plot(sub["Date"], sub["True"], label="True", color=TRUE_COLOR, linewidth=TRUE_WIDTH)
                    axs[j].plot(sub["Date"], sub["Pred"], label="Predicted", color=PRED_COLOR, linewidth=PRED_WIDTH)
                    axs[j].set_title(f"{well} - t+{h}")
                    axs[j].set_xlabel("Date")
                    axs[j].set_ylabel("Water Level")
                    axs[j].tick_params(axis='x', rotation=45)
                    axs[j].xaxis.set_major_locator(plt.MaxNLocator(6))
                    axs[j].legend()
                plt.suptitle(f"{well} Forecast (t+{HORIZONS[pair[0]]} / t+{HORIZONS[pair[1]]})", fontsize=14)
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                pdf.savefig()
                plt.close()

    print(f"\n[Done] LSTM + LAG Report saved to: {pdf_path}")

# ====== 실행 ======
if __name__ == '__main__':
    run_lstm_pipeline()
