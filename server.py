from typing import List, Tuple
from flwr.common import Metrics, ndarrays_to_parameters
from flwr.server import start_server, ServerConfig
from flwr.server.strategy import FedAdam
from task import load_model

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

def server_fn(fractional_fit, num_rounds):
    """Construct components that set the ServerApp behaviour."""
    # Let's define the global model and pass it to the strategy
    parameters = ndarrays_to_parameters(load_model(learning_rate=0.001, input_shape=(32,32,3), datatype="Image").get_weights())
    # Define the strategy
    strategy = FedAdam(
        fraction_fit=fractional_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
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
    