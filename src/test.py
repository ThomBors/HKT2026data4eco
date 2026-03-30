import torch
import numpy as np
import matplotlib.pyplot as plt


def run_test(
    df_weather_test,
    model_path="saved_models/windPrc-cfrnn-42.pt",
    length=48,
    horizon=24,
    n_plots=1
):
    # 1️⃣ Load model
    model = torch.load(model_path, map_location="cpu")
    model.eval()

    # 2️⃣ Prepare data
    X_raw = df_weather_test.values  # (T, n_features)

    # 3️⃣ Build sequences
    n_samples = len(X_raw) - length + 1
    X_test = np.array([X_raw[i:i+length] for i in range(n_samples)])

    X_tensor = torch.FloatTensor(X_test)

    # 4️⃣ Predict
    with torch.no_grad():
        intervals, _ = model.predict(X_tensor)

    lower = intervals[:, 0].numpy()
    upper = intervals[:, 1].numpy()
    preds = (lower + upper) / 2

    # 5️⃣ Plot random samples
    for _ in range(n_plots):
        i = np.random.randint(0, len(X_test))

        sequence = X_test[i]
        pred = preds[i]
        low = lower[i]
        up = upper[i]

        # ⚠️ using first feature as proxy (adjust if needed)
        last_val = sequence[-1, 0]

        pred_line = [last_val] + pred.flatten().tolist()
        low_line = [last_val] + low.flatten().tolist()
        up_line = [last_val] + up.flatten().tolist()

        plt.figure(figsize=(12, 5))

        # past
        plt.plot(range(1, length + 1), sequence[:, 0], label="Input (feature 0)")

        # forecast
        plt.plot(range(length, length + horizon + 1), pred_line, linestyle="--", label="Prediction")
        plt.fill_between(range(length, length + horizon + 1), low_line, up_line, alpha=0.3, label="Interval")

        plt.axvline(length, linestyle="--")

        plt.title("CFRNN Forecast on Test Set")
        plt.xlabel("Time step")
        plt.ylabel("Value")
        plt.legend()
        plt.show()

    return preds, lower, upper