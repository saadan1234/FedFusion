from flwr.client import start_client, NumPyClient
from task import load_data, load_model
from time import localtime, strftime




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
        print(f"{strftime("%Y-%m-%d %H:%M:%S", localtime())} Flower Client Constructor")
        self.model = load_model(learning_rate)
        self.x_train, self.y_train, self.x_test, self.y_test = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        print(f"{strftime("%Y-%m-%d %H:%M:%S", localtime())} Starting Fit Function: Parameters: {type(parameters)}")
        self.model.set_weights(parameters)
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )
        print(f"{strftime("%Y-%m-%d %H:%M:%S", localtime())} Ending Fit Function: Weights: {type(self.model.get_weights())} \nLength: {len(self.model.get_weights())}")
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        print(f"{strftime("%Y-%m-%d %H:%M:%S", localtime())} Starting Evaluate Function: Parameters: {type(parameters)}")
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f"{strftime("%Y-%m-%d %H:%M:%S", localtime())} Ending Evaluate Function: loss: {type(loss)} \nAccuracy: {type(accuracy)}")
        return loss, len(self.x_test), {"accuracy": accuracy}


def client_fn(partition_id, num_partitions, local_epochs, batch_size, learning_rate):
    """Construct a Client that will be run in a ClientApp."""
    print(f"{strftime("%Y-%m-%d %H:%M:%S", localtime())} Starting Client Function 1")
    # Read the node_config to fetch data partition associated to this node
    partition_id = partition_id
    num_partitions = num_partitions
    data = load_data(partition_id, num_partitions)

    # Read run_config to fetch hyperparameters relevant to this run
    epochs = local_epochs
    batch_size = batch_size
    verbose = 1
    learning_rate = learning_rate

    print(f"{strftime("%Y-%m-%d %H:%M:%S", localtime())} Returning Client 1")
    # Return Client instance
    return FlowerClient(learning_rate, data, epochs, batch_size, verbose).to_client()


if __name__ == "__main__":
    print(f"{strftime("%Y-%m-%d %H:%M:%S", localtime())} Starting Client 1")
    client = client_fn(partition_id=0, num_partitions=2, local_epochs=2, batch_size=32, learning_rate=0.1)
    start_client(
        server_address='127.0.0.1:8080',
        client=client
    )