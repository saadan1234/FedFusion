from typing import List, Tuple

from flwr.common import Metrics, ndarrays_to_parameters
from flwr.server import start_server, ServerConfig
from flwr.server.strategy import FedAdam
from task import load_model
from time import localtime, strftime


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    print(f"{strftime("%Y-%m-%d %H:%M:%S", localtime())} Entering Averaging Function")
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    print(f"{strftime("%Y-%m-%d %H:%M:%S", localtime())} Averaging Function: Accuracies: {type(accuracies)} \nLength: {len(accuracies)}")
    print(f"{strftime("%Y-%m-%d %H:%M:%S", localtime())} Averaging Function: Parameters: {type(examples)} \nLength: {len(examples)}")
    # Aggregate and return custom metric (weighted average)
    print(f"{strftime("%Y-%m-%d %H:%M:%S", localtime())} Averaging Function: Metrics: {type(metrics)} \nLength: {len(metrics)}")
    return {"accuracy": sum(accuracies) / sum(examples)}


def server_fn(fractional_fit, num_rounds):
    """Construct components that set the ServerApp behaviour."""
    print(f"{strftime("%Y-%m-%d %H:%M:%S", localtime())} Starting Server Function 1")
    # Let's define the global model and pass it to the strategy
    parameters = ndarrays_to_parameters(load_model().get_weights())
    print(f"{strftime("%Y-%m-%d %H:%M:%S", localtime())} Server Function: Parameters: {type(parameters)}")
    # Define the strategy
    strategy = FedAdam(
        fraction_fit=fractional_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    print(f"{strftime("%Y-%m-%d %H:%M:%S", localtime())} Server Function: Weights: {type(load_model().get_weights())} \nLength: {len(load_model().get_weights())}")
    # Read from config
    config = ServerConfig(num_rounds=num_rounds)

    return config, strategy


if __name__ == "__main__":
    config, strategy = server_fn(fractional_fit=0.5, num_rounds=3)
    start_server(
        server_address='0.0.0.0:8080',
        config=config,
        strategy=strategy
    )