from flwr.client import Client
from flwr.common import (
    Parameters,
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
    GetParametersIns,
    GetParametersRes,
    Status,
    Code
)
from clientutils import build_model  # Function to build the required ML model
from crypto.rsa_crypto import RsaCryptoAPI  # Handles RSA encryption/decryption
import tensorflow as tf

def create_gradient_leakage_client(input_shape, num_classes, model_type, X_train, Y_train, X_test, Y_test):
    """
    Creates a Flower client that simulates gradient leakage attacks.

    Args:
        input_shape: Shape of input data.
        num_classes: Number of output classes.
        model_type: Type of model (e.g., dense, LSTM).
        X_train: Training data.
        Y_train: Training labels.
        X_test: Testing data.
        Y_test: Testing labels.

    Returns:
        An instance of `GradientLeakageClient`.
    """
    
    class GradientLeakageClient(Client):
        """
        A custom Flower client that:
        1. Implements gradient leakage attack simulation.
        2. Handles encrypted communication of model weights.
        """
        
        def __init__(self):
            """
            Initializes the client:
            - Loads AES key for encryption/decryption.
            - Builds the specified ML model.
            """
            super().__init__()
            self.aes_key = self.load_key("crypto/aes_key.bin")
            self.model = build_model(input_shape, num_classes, model_type)

        @staticmethod
        def load_key(filename):
            """
            Loads an AES encryption key from a file.
            
            Args:
                filename: Path to the key file.

            Returns:
                The AES key in binary format.
            """
            with open(filename, "rb") as f:
                return f.read()

        def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
            """
            Encrypts and returns the model's parameters.
            
            Args:
                ins: Request to get model parameters.

            Returns:
                Encrypted model parameters.
            """
            print("Getting model parameters for encryption.")
            enc_params = [
                RsaCryptoAPI.encrypt_numpy_array(self.aes_key, w)
                for w in self.model.get_weights()
            ]
            print(f"Encrypted parameters: {[len(param) for param in enc_params]}")
            return GetParametersRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=Parameters(tensors=enc_params, tensor_type=""),
            )

        def set_parameters(self, parameters: Parameters):
            """
            Decrypts and sets the model's parameters.
            
            Args:
                parameters: Encrypted model parameters.
            """
            dec_params = [
                RsaCryptoAPI.decrypt_numpy_array(
                    self.aes_key, param, dtype=self.model.get_weights()[i].dtype
                ).reshape(self.model.get_weights()[i].shape)
                for i, param in enumerate(parameters.tensors)
            ]
            self.model.set_weights(dec_params)

        def fit(self, ins: FitIns) -> FitRes:
            """
            Trains the model and performs a gradient leakage attack.
            
            Args:
                ins: Instructions containing encrypted model parameters.

            Returns:
                Updated model parameters after training, with metrics including similarity score.
            """
            # Set the model parameters
            self.set_parameters(ins.parameters)

            # Train the model
            self.model.fit(X_train, Y_train, epochs=1, batch_size=32, verbose=1)

            # Perform a gradient leakage attack on a single sample
            print("Performing Gradient Leakage Attack...")
            optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

            # Prepare a single batch of data
            inputs = tf.convert_to_tensor(X_train[:1], dtype=tf.float32)
            targets = tf.convert_to_tensor(Y_train[:1], dtype=tf.int32)

            with tf.GradientTape() as tape:
                tape.watch(inputs)
                outputs = self.model(inputs)
                loss = loss_fn(targets, outputs)

            # Capture gradients
            gradients = tape.gradient(loss, self.model.trainable_weights)

            # Perform the attack
            reconstructed_data, similarity_score = gradient_leakage_attack_keras(
                self.model, gradients, inputs.shape, inputs.numpy()
            )

            # Log similarity results
            print(f"Similarity Score (MSE) between actual and reconstructed data: {similarity_score}")

            # Return updated model weights
            get_param_ins = GetParametersIns(config={"aes_key": self.aes_key})
            return FitRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=self.get_parameters(get_param_ins).parameters,
                num_examples=len(X_train),
                metrics={"similarity_score": float(similarity_score)},  # Cast to native float
            )

        def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
            """
            Evaluates the model using test data.
            
            Args:
                ins: Instructions containing encrypted model parameters.

            Returns:
                Evaluation results, including loss and accuracy.
            """
            print("Decrypting model parameters for evaluation.")
            self.set_parameters(ins.parameters)
            loss, accuracy = self.model.evaluate(X_test, Y_test)
            print(f"Evaluation results - Loss: {loss}, Accuracy: {accuracy}")
            return EvaluateRes(
                status=Status(code=Code.OK, message="Success"),
                loss=loss,
                num_examples=len(X_test),
                metrics={"accuracy": accuracy},
            )

    return GradientLeakageClient()

def gradient_leakage_attack_keras(model, gradients, target_shape, actual_data):
    """
    Simulates a gradient leakage attack to reconstruct input data from gradients.

    Args:
        model: The Keras model whose gradients are leaked.
        gradients: The leaked gradients.
        target_shape: Shape of the original data.
        actual_data: The actual data for similarity comparison.

    Returns:
        reconstructed_data: Reconstructed data from gradients.
        similarity_score: Measure of similarity between reconstructed and actual data.
    """
    # Initialize random data for optimization
    reconstructed_data = tf.Variable(tf.random.normal(target_shape), trainable=True)

    # Use SGD to reconstruct data
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    loss_fn = tf.keras.losses.MeanSquaredError()

    for step in range(10):  # Tune steps for precision/speed tradeoff
        with tf.GradientTape(persistent=True) as tape:  # Persistent tape for multiple gradient computations
            tape.watch(reconstructed_data)
            outputs = model(reconstructed_data)
            # Match the gradients
            loss = tf.add_n([
                loss_fn(g, r)
                for g, r in zip(gradients, tape.gradient(outputs, model.trainable_weights))
            ])

        # Apply gradients to reconstructed data
        grads = tape.gradient(loss, [reconstructed_data])
        optimizer.apply_gradients(zip(grads, [reconstructed_data]))

        if step % 2 == 0:
            print(f"Step {step}, Loss: {loss.numpy()}")

    del tape  # Explicitly delete tape to free memory

    # Convert to numpy and calculate similarity
    reconstructed_data_np = reconstructed_data.numpy()
    similarity_score = tf.reduce_mean(tf.square(actual_data - reconstructed_data_np)).numpy()

    return reconstructed_data_np, similarity_score
