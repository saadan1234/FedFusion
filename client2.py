from flwr.client import start_client, NumPyClient
from task import load_data, load_model


# Define Flower Client
class FlowerClient(NumPyClient):
    def __init__(
        self,
        learning_rate,
        data,
        epochs,
        batch_size,
        verbose,
    ):
        print("Flower Client Constructor")
        self.model = load_model(learning_rate)
        self.x_train, self.y_train, self.x_test, self.y_test = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        print(f"Starting Fit Function 2: Parameters: {type(parameters)} {parameters}")
        self.model.set_weights(parameters)
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )
        print(f"Ending Fit Function 2: Weights: {type(self.model.get_weights())} \nLength: {len(self.model.get_weights())} {self.model.get_weights()}")
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        print(f"Starting Evaluate Function 2: Parameters: {type(parameters)} {parameters}")
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f"Ending Evaluate Function 2: loss: {type(loss)} {loss} \nAccuracy: {type(accuracy)} {accuracy}")
        return loss, len(self.x_test), {"accuracy": accuracy}


def client_fn(partition_id, num_partitions, local_epochs, batch_size, learning_rate):
    """Construct a Client that will be run in a ClientApp."""
    print("Starting Client Function 2")
    # Read the node_config to fetch data partition associated to this node
    partition_id = partition_id
    num_partitions = num_partitions
    data = load_data(partition_id, num_partitions)

    # Read run_config to fetch hyperparameters relevant to this run
    epochs = local_epochs
    batch_size = batch_size
    verbose = 1
    learning_rate = learning_rate

    print("Returning Client 2")
    # Return Client instance
    return FlowerClient(learning_rate, data, epochs, batch_size, verbose).to_client()


if __name__ == "__main__":
    print("Starting Client 2")
    client = client_fn(partition_id=0, num_partitions=2, local_epochs=2, batch_size=32, learning_rate=0.1)
    start_client(
        server_address='127.0.0.1:8080',
        client=client
    )