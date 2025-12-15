import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.tree import DecisionTreeRegressor
from abc import ABC, abstractmethod

class LearnedIndex(ABC):
    """
    Abstract base class for a learned index.
    Provides a generic search method and defines the interface for model training and prediction.
    """
    def train(self, sorted_z_values: list[int], progress_callback=None):
        self.training_function(sorted_z_values, progress_callback)
        self.max_error = self.error_bound(sorted_z_values)

    @abstractmethod
    def training_function(self, sorted_z_values: list[int], progress_callback=None):
        """Trains the underlying machine learning model."""
        pass

    @abstractmethod
    def predict(self, z_value: int) -> int:
        """Uses the trained model to predict the index for a given z_value."""
        pass

    def error_bound(self, sorted_z_values: list[int]) -> float:
        """
        Computes the absolute maximum prediction error bounds after training.
        Returns the maximum absolute error between predicted and actual indices.
        
        This error bound is used to determine the search range during queries.
        A smaller error bound means more accurate predictions and potentially
        fewer comparisons needed during search.
        """
        # TODO: Implement maximum absolute error bound calculation

    def search(self, sorted_z_values: list[int], query: int) -> tuple[int, int]:
        """
        Performs a hybrid search using the learned model.
        
        Strategy:
        1. Predict the position of the query using the learned model
        2. Clamp the prediction to valid array bounds
        3. Check if we got lucky and the prediction is exact
        4. Otherwise, search within the error bounds range around the prediction or perform an exponential search starting from the predicted position
        
        Returns:
            A tuple of (index of found element or -1 if not found, number of comparisons made).
        """
        n = len(sorted_z_values)
        if n == 0: return -1, 0
        comparisons = 0

        # TODO: Implement prediction of the key position
        # Make sure the predicted position is within bounds.

        # TODO: Either use error bounds or exponential search starting from the predicted position
        # Count number of comparisons made!

        # Return -1 if not found, along with comparisons made
        return -1, comparisons

class PyTorchModel(LearnedIndex):
    """
    Abstract base class for PyTorch-based learned index models.
    Handles the training loop.
    """
    def __init__(self, model: nn.Module, epochs=10, lr=0.01, batch_size=2048, show_progress=False):
        super().__init__()
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.n = None
        self.x_min = None
        self.x_range = None

    def training_function(self, sorted_z_values: list[int], progress_callback=None):
        n = len(sorted_z_values)
        if n < 2: return
        self.n = n

        X = torch.tensor(sorted_z_values, dtype=torch.float32).view(-1, 1)
        y = torch.tensor(range(n), dtype=torch.float32).view(-1, 1)

        self.x_min = X.min()
        self.x_range = X.max() - self.x_min

        X = (X - self.x_min) / self.x_range
        y = y / n
        
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        
        if self.show_progress:
            try:
                from tqdm import tqdm
                epoch_iterator = tqdm(range(self.epochs), desc="Training", unit="epoch")
            except ImportError:
                epoch_iterator = range(self.epochs)
        else:
            epoch_iterator = range(self.epochs)
        
        for epoch in epoch_iterator:
            epoch_loss = 0.0
            num_batches = 0
            for inputs, targets in loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1

            # Update progress callback if provided (for Streamlit)
            if progress_callback is not None:
                progress_callback(epoch + 1, self.epochs, epoch_loss / num_batches)
            
            # Update tqdm description with loss
            if self.show_progress and hasattr(epoch_iterator, 'set_postfix'):
                epoch_iterator.set_postfix({'loss': f'{epoch_loss / num_batches:.4f}'})

    def predict(self, z_value: int) -> int:
        self.model.eval()
        with torch.no_grad():
            # TODO: Implement prediction using the pytorch model
            # HINT: Remember to normalize the input z_value and denormalize the output
            pass

class PyTorchLinearModel(PyTorchModel):
    """A learned index using a single-layer linear PyTorch model."""
    def __init__(self, epochs=10, lr=0.01, batch_size=2048, show_progress=False):
        # Define the simple linear model architecture
        model = nn.Linear(1, 1)
        super().__init__(model=model, epochs=epochs, lr=lr, batch_size=batch_size, show_progress=show_progress)

class PyTorchMLPModel(PyTorchModel):
    """A learned index using a Multi-Layer Perceptron PyTorch model."""
    def __init__(self, epochs=10, lr=0.001, batch_size=2048, show_progress=False):
        # Define the MLP architecture
        model = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        super().__init__(model=model, epochs=epochs, lr=lr, batch_size=batch_size, show_progress=show_progress)

class DecisionTreeModel(LearnedIndex):
    """
    A learned index implementation using a Decision Tree regressor
    from scikit-learn.
    """

    def __init__(self):
        super().__init__()
        self.model = DecisionTreeRegressor(random_state=1, max_depth=20)
        self.n = None

    def training_function(self, sorted_z_values: list[int], progress_callback=None):
        n = len(sorted_z_values)
        if n < 2:
            print("Not enough data to train.")
            return

        X = np.array(sorted_z_values).reshape(-1, 1)
        y = np.arange(n) / n
        self.n = n

        self.model.fit(X, y)

    def predict(self, z_value: int) -> int:
        # TODO: Implement prediction using the decision tree model
        pass
