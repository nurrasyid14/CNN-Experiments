import numpy as np
import torch

from learning.models.lstm import CNNLSTMHead
from learning.models.mlp import CNNMLPHead
from learning.models.rnn import CNNRNNHead
from learning.training.head_trainer import train_embedding_head, predict_embedding_head


def test_mlp_head_output_shape() -> None:
    features = torch.randn(4, 16)
    model = CNNMLPHead(input_dim=16, num_classes=3)
    assert model(features).shape == (4, 3)


def test_lstm_head_output_shape() -> None:
    features = torch.randn(4, 16)
    model = CNNLSTMHead(input_dim=16, num_classes=3, sequence_length=4)
    assert model(features).shape == (4, 3)


def test_rnn_head_output_shape() -> None:
    features = torch.randn(4, 16)
    model = CNNRNNHead(input_dim=16, num_classes=3, sequence_length=4)
    assert model(features).shape == (4, 3)


def test_embedding_head_training_and_prediction() -> None:
    features = np.vstack(
        [np.zeros((8, 16)), np.ones((8, 16))]
    ).astype(np.float32)
    labels = np.repeat([0, 1], 8).astype(np.int64)
    model = CNNMLPHead(16, 2)
    history = train_embedding_head(
        model,
        features,
        labels,
        features[:4],
        labels[:4],
        torch.device("cpu"),
        epochs=1,
    )
    predictions, probabilities = predict_embedding_head(
        model, features[:3], torch.device("cpu")
    )
    assert len(history) == 1
    assert predictions.shape == (3,)
    assert probabilities.shape == (3, 2)
