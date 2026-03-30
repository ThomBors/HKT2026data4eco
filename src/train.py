import pickle

import torch

from model.cfrnn import CFRNN
from data.dataloader import get_splits
from utils.perfomance import evaluate_cfrnn_performance

DEFAULT_PARAMETERS = {
    "batch_size": 150,
    "embedding_size": 20,
    "coverage": 0.9,
    "lr": 0.01,
    "n_steps": 1000,
    "input_size": 1,
    "rnn_mode": "LSTM",
}


def run_experiments(params=None, save_model=False, save_results=True, seed=42):
       
    horizon = 50
    length = 100

    # Parameters
    params = DEFAULT_PARAMETERS.copy() if params is None else params
    params["max_steps"] = length
    params["output_size"] = horizon
    params["epochs"] = 100

    torch.manual_seed(seed)

    print("Training {}".format('cfrnn'))

    
    train_dataset, calibration_dataset, test_dataset = get_splits(conformal=True, horizon=horizon, seed=seed)

    model = CFRNN(
        embedding_size=params["embedding_size"],
        horizon=50,
        error_rate=1 - params["coverage"],
        mode=params["rnn_mode"],
    )

    model.fit(
        train_dataset,
        calibration_dataset,
        epochs=params["epochs"],
        lr=params["lr"],
        batch_size=params["batch_size"],
    )

    results = evaluate_cfrnn_performance(model, test_dataset)

    if save_model:
        torch.save(model, "saved_models/{}-{}-{}.pt".format('windPrc', 'cfrnn', seed))
    if save_results:
        with open("../results/{}-{}-{}.pkl".format('windPrc', 'cfrnn', seed), "wb") as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    return results


if __name__ == "__main__":
    run_experiments()