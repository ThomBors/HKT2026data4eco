import pickle

import torch

from model.cfrnn import CFRNN
from data.dataloader import get_splits
from utils.perfomance import evaluate_cfrnn_performance, predict_on_test,save_predictions_csv

DEFAULT_PARAMETERS = {
    "batch_size": 256,
    "embedding_size": 80,
    "coverage": 0.9,
    "lr": 0.01,
    "n_steps": 1000,
    "input_size": 125,
    "output_size": 1,
    "rnn_mode": "LSTM",
}



def run_experiments(params=None, save_model=True, save_results=True, seed=42):
       
    horizon = 1
    length = 96

    # Parameters
    params = DEFAULT_PARAMETERS.copy() if params is None else params
    params["max_steps"] = length
    params["output_size"] = horizon
    params["epochs"] = 1000

    torch.manual_seed(seed)

    print("Training {}".format('cfrnn'))

    
    #train_dataset, calibration_dataset, test_dataset = get_splits(conformal=True, horizon=horizon, seed=seed)
    train_dataset, val_dataset, calibration_dataset, test_dataset = get_splits( horizon=horizon, seed=seed)

    model = CFRNN(
        input_size=params['input_size'],
        embedding_size=params["embedding_size"],
        horizon=horizon,
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

    results = evaluate_cfrnn_performance(model, val_dataset)

    if save_model:
        torch.save(model, "saved_models/{}-{}-{}.pt".format('windPrc', 'cfrnn', seed))
    if save_results:
        with open("results/{}-{}-{}.pkl".format('windPrc', 'cfrnn', seed), "wb") as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    x, lower, upper, preds = predict_on_test(model, test_dataset)

    save_predictions_csv(x, lower, upper, preds,path="results/predictions.csv")




if __name__ == "__main__":
    run_experiments()