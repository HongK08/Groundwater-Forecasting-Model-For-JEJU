import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# 수정된 NBeatsNet 클래스
# -----------------------------
class NBeatsBlock(nn.Module):
    def __init__(self, input_size, theta_size, hidden_size, n_layers):
        super().__init__()
        self.hidden = nn.ModuleList([
            nn.Linear(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(n_layers)
        ])
        self.theta = nn.Linear(hidden_size, theta_size)

    def forward(self, x):
        for layer in self.hidden:
            x = F.relu(layer(x))
        return self.theta(x)

class NBeatsNet(nn.Module):
    def __init__(self, input_size, forecast_size, hidden_size=128, n_layers=4, stack_types=2, blocks_per_stack=3):
        super().__init__()
        self.input_size = input_size
        self.forecast_size = forecast_size

        self.stacks = nn.ModuleList()
        for _ in range(stack_types):
            stack = nn.ModuleList([
                NBeatsBlock(input_size, input_size + forecast_size, hidden_size, n_layers)
                for _ in range(blocks_per_stack)
            ])
            self.stacks.append(stack)

    def forward(self, x):
        device = x.device
        x = x.view(x.size(0), -1)
        backcast = torch.zeros((x.size(0), self.input_size), device=device)
        forecast = torch.zeros((x.size(0), self.forecast_size), device=device)

        for stack in self.stacks:
            for block in stack:
                theta = block(x)
                backcast_theta = theta[:, :self.input_size]
                forecast_theta = theta[:, self.input_size:]
                backcast = backcast + backcast_theta
                forecast = forecast + forecast_theta
                x = x - backcast_theta

        return backcast, forecast

# -----------------------------
# 설정 (전역 하이퍼파라미터)
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
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 128
N_LAYERS = 4
STACK_TYPES = 2
BLOCKS_PER_STACK = 3

# -----------------------------
# Dataset 정의
# -----------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, window_size, horizon):
        self.X = X
        self.y = y
        self.window_size = window_size
        self.horizon = horizon

    def __len__(self):
        return len(self.X) - self.window_size - self.horizon

    def __getitem__(self, idx):
        return (
            self.X[idx:idx + self.window_size],
            self.y[idx + self.window_size:idx + self.window_size + self.horizon]
        )



# -----------------------------
# 평가 함수
# -----------------------------
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2

# -----------------------------
# 메인 실행
# -----------------------------
def main():
    global FEATURE_COLS
    df = pd.read_csv('/media/user/AI_2T/UML_Paper/Main_Data_Jeju/JeJu_merged.csv')
    for col in TARGET_COLS:
        for lag in [1, 2, 3]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').dropna().reset_index(drop=True)

    FEATURE_COLS += [f"{col}_lag{lag}" for col in TARGET_COLS for lag in [1, 2, 3]]
    df = df.dropna().reset_index(drop=True)
    X_all = df[FEATURE_COLS].values
    y_all = df[TARGET_COLS].values

    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_scaled = X_scaler.fit_transform(X_all)
    y_scaled = y_scaler.fit_transform(y_all)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results = {}
    pred_df_rows = []
    for horizon in HORIZONS:
        window_size = WINDOW_MAP[horizon]
        batch_size = BATCH_SIZE
        epochs = EPOCHS

        for target_idx, target_col in enumerate(TARGET_COLS):
            dataset = TimeSeriesDataset(X_tensor, y_tensor[:, target_idx], window_size, horizon)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            model = NBeatsNet(
                input_size=window_size * X_tensor.shape[1],
                forecast_size=horizon,
                hidden_size=HIDDEN_SIZE,
                n_layers=N_LAYERS,
                stack_types=STACK_TYPES,
                blocks_per_stack=BLOCKS_PER_STACK
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            loss_fn = nn.MSELoss()

            model.train()
            for epoch in range(epochs):
                xb_all, yb_all = [], []
                for xb, yb in dataloader:
                    xb_all.append(xb.view(xb.size(0), -1))
                    yb_all.append(yb)
                xb_all = torch.cat(xb_all).to(device)
                yb_all = torch.cat(yb_all).to(device)
                for i in range(0, xb_all.size(0), batch_size):
                    xb_batch = xb_all[i:i+batch_size]
                    yb_batch = yb_all[i:i+batch_size]
                    optimizer.zero_grad()
                    backcast, forecast = model(xb_batch)
                    loss = loss_fn(forecast, yb_batch)
                    loss.backward()
                    optimizer.step()

            # 예측
            model.eval()
            X_inputs = torch.stack([
                X_tensor[i:i+window_size].reshape(-1)
                for i in range(len(X_tensor) - window_size - horizon)
            ])
            X_inputs = X_inputs.to(device)
            with torch.no_grad():
                _, forecasts = model(X_inputs)

            y_pred_all, y_true_all = [], []
            for i in range(len(forecasts)):
                forecast = forecasts[i].unsqueeze(0)
                y_pred_all.append(forecast.cpu().numpy().flatten())
                y_true = y_tensor[i + window_size:i + window_size + horizon, target_idx].numpy()
                y_true_all.append(y_true)

                # 역변환된 예측 및 실제값 평균 저장
                y_pred_point = forecast.cpu().numpy().flatten().mean()
                y_true_point = y_true.mean()
                pred_full = np.zeros((1, len(TARGET_COLS)))
                true_full = np.zeros((1, len(TARGET_COLS)))
                pred_full[0, target_idx] = y_pred_point
                true_full[0, target_idx] = y_true_point
                y_pred_inv = y_scaler.inverse_transform(pred_full)[0, target_idx]
                y_true_inv = y_scaler.inverse_transform(true_full)[0, target_idx]
                date_idx = i + window_size + horizon
                pred_df_rows.append({
                    "Well": target_col,
                    "Horizon": f"t+{horizon}",
                    "Date": df["date"].iloc[date_idx],
                    "True": y_true_inv,
                    "Pred": y_pred_inv
                })

                # 역변환된 예측 및 실제값 평균 저장
                y_pred_point = forecast.cpu().numpy().flatten().mean()
                y_true_point = y_true.mean()
                pred_full = np.zeros((1, len(TARGET_COLS)))
                true_full = np.zeros((1, len(TARGET_COLS)))
                pred_full[0, target_idx] = y_pred_point
                true_full[0, target_idx] = y_true_point
                y_pred_inv = y_scaler.inverse_transform(pred_full)[0, target_idx]
                y_true_inv = y_scaler.inverse_transform(true_full)[0, target_idx]
                date_idx = i + window_size + horizon
                pred_df_rows.append({
                    "Well": target_col,
                    "Horizon": f"t+{horizon}",
                    "Date": df["date"].iloc[date_idx],
                    "True": y_true_inv,
                    "Pred": y_pred_inv
                })

            y_pred_all = np.array(y_pred_all)
            y_true_all = np.array(y_true_all)

            # 역변환
            y_pred_full = np.zeros((y_pred_all.shape[0], len(TARGET_COLS)))
            y_true_full = np.zeros((y_true_all.shape[0], len(TARGET_COLS)))
            y_pred_full[:, target_idx] = y_pred_all.mean(axis=1)
            y_true_full[:, target_idx] = y_true_all.mean(axis=1)
            y_pred = y_scaler.inverse_transform(y_pred_full)[:, target_idx]
            y_true = y_scaler.inverse_transform(y_true_full)[:, target_idx]

            mae, mse, rmse, r2 = evaluate(y_true, y_pred)
            results[f"{target_col},t+{horizon}"] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

    # 결과 저장
    result_df = pd.DataFrame(results).T
    result_df.to_csv("/media/user/AI_2T/UML_Paper/Daytime/nbeats/nbeats_prediction_results.csv")

    # PDF 리포트 저장
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.backends.backend_pdf import PdfPages
    os.makedirs("/media/user/AI_2T/UML_Paper/Daytime/nbeats/plots", exist_ok=True)
    pdf_path = "/media/user/AI_2T/UML_Paper/Daytime/nbeats/plots/nbeats_report.pdf"
    summary_df = result_df.reset_index()
    summary_df.columns = ['Key', 'MAE', 'MSE', 'RMSE', 'R2']
    summary_df[['Target', 'Horizon']] = summary_df['Key'].str.split(',', expand=True)
    summary_df = summary_df[['Target', 'Horizon', 'MAE', 'MSE', 'RMSE', 'R2']]

    mae_matrix = summary_df.pivot(index='Target', columns='Horizon', values='MAE')
    mse_matrix = summary_df.pivot(index='Target', columns='Horizon', values='MSE')
    rmse_matrix = summary_df.pivot(index='Target', columns='Horizon', values='RMSE')
    r2_matrix = summary_df.pivot(index='Target', columns='Horizon', values='R2')

    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(12, 0.5 + len(summary_df)*0.25))
        ax.axis('off')
        tbl = ax.table(cellText=summary_df.round(4).values, colLabels=summary_df.columns,
                       loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.0, 1.5)
        pdf.savefig(); plt.close()

        for name, matrix in zip(['MSE', 'RMSE', 'R2'], [mse_matrix, rmse_matrix, r2_matrix]):
            plt.figure(figsize=(10, 6))
            sns.heatmap(matrix.astype(float), annot=True, fmt=".3f", cmap="YlGnBu")
            plt.title(f"{name} Heatmap (N-BEATS)")
            plt.xlabel("Horizon")
            plt.ylabel("Well")
            plt.tight_layout()
            pdf.savefig(); plt.close()

        pred_df = pd.DataFrame(pred_df_rows)
        pred_df.to_csv("/media/user/AI_2T/UML_Paper/Daytime/nbeats/nbeats_predictions.csv", index=False)

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
                pdf.savefig()
                plt.close()

            print(f"=[Done] N-BEATS Report saved to: {pdf_path}")

if __name__ == '__main__':
    main()
