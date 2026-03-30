import pandas as pd
import numpy as np
import torch



def evaluate_cfrnn_performance(model, test_dataset, correct_conformal=True):
    independent_coverages, joint_coverages, intervals = model.evaluate_coverage(
        test_dataset, corrected=correct_conformal
    )
    mean_independent_coverage = torch.mean(independent_coverages.float(), dim=0)
    mean_joint_coverage = torch.mean(joint_coverages.float(), dim=0).item()
    interval_widths = (intervals[:, 1] - intervals[:, 0]).squeeze()
    point_predictions, errors = model.get_point_predictions_and_errors(test_dataset, corrected=correct_conformal)

    results = {
        "Point predictions": point_predictions,
        "Errors": errors,
        "Independent coverage indicators": independent_coverages.squeeze(),
        "Joint coverage indicators": joint_coverages.squeeze(),
        "Upper limit": intervals[:, 1],
        "Lower limit": intervals[:, 0],
        "Mean independent coverage": mean_independent_coverage.squeeze(),
        "Mean joint coverage": mean_joint_coverage,
        "Confidence interval widths": interval_widths,
        "Mean confidence interval widths": interval_widths.mean(dim=0),
    }

    return results


def predict_on_test(model, test_dataset, corrected=True):
    model.auxiliary_forecaster.eval()

    all_lower = []
    all_upper = []
    all_preds = []


    with torch.no_grad():
        for sequences, targets, lengths in test_dataset:
            # Point predictions
            preds, _ = model.auxiliary_forecaster(sequences)

            # Intervals
            intervals, _ = model.predict(sequences, corrected=corrected)

            lower = intervals[:, 0]   # [batch, horizon, 1]
            upper = intervals[:, 1]

            all_lower.append(lower)
            all_upper.append(upper)
            all_preds.append(preds)

    lower = torch.cat(all_lower)   # [N, horizon, 1]
    upper = torch.cat(all_upper)
    preds = torch.cat(all_preds)

    # Simple x-axis (sample index)
    x = torch.arange(lower.shape[0])

    return x, lower, upper, preds





def save_predictions_csv(x, lower, upper, preds, path="results/predictions.csv"):
    # Convert to numpy
    lower = lower.squeeze(-1).cpu().numpy()   # [N, horizon]
    upper = upper.squeeze(-1).cpu().numpy()
    preds = preds.squeeze(-1).cpu().numpy()

    N, H = lower.shape

    rows = []

    for i in range(N):
        for h in range(H):
            rows.append({
                "sample": i,
                "horizon_step": h,
                "prediction": preds[i, h],
                "lower": lower[i, h],
                "upper": upper[i, h],
            })

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)

    print(f"Saved to {path}")




def compute_Ef(y_true: np.ndarray, y_pred: np.ndarray, freq_per_day: int = 24):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_days = len(y_true) // freq_per_day
    daily_abs_error = []
    daily_bias = []
    
    for d in range(n_days):
        start = d * freq_per_day
        end = start + freq_per_day
        true_day = y_true[start:end]
        pred_day = y_pred[start:end]
        
        eps_abs = np.sum(np.abs(pred_day - true_day))
        eps_bias = np.sum(pred_day - true_day)
        
        daily_abs_error.append(eps_abs)
        daily_bias.append(eps_bias)
    
    daily_abs_error = np.array(daily_abs_error)
    daily_bias = np.array(daily_bias)
    
    Ef = np.sum(daily_abs_error) + np.abs(np.sum(daily_bias))
    return Ef, daily_abs_error, daily_bias