
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


