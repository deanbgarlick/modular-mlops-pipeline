"""Simple PyTorch neural network model implementation (original version)."""

import numpy as np
from typing import Any, Optional
import warnings

from .base import Model


class SimplePyTorchNeuralNetwork(Model):
    """Simple PyTorch neural network model for binary classification (original version)."""
    
    def __init__(self, hidden_size: int = 128, learning_rate: float = 0.001, 
                 epochs: int = 100, batch_size: int = 32, random_state: int = 42):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.model = None
        self.device = None
        self.is_fitted = False
        self.input_size = None
        self.classes_ = None
        
    def _check_pytorch_available(self):
        """Check if PyTorch is available."""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
            return torch, nn, optim, DataLoader, TensorDataset
        except ImportError:
            raise ImportError("PyTorch is required for neural network model. Install with: pip install torch")
    
    def _create_model(self, input_size: int):
        """Create the neural network architecture."""
        torch, nn, _, _, _ = self._check_pytorch_available()
        
        class BinaryClassifier(nn.Module):
            def __init__(self, input_size, hidden_size):
                super(BinaryClassifier, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
                self.fc3 = nn.Linear(hidden_size // 2, 2)  # Binary classification
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.2)
                self.sigmoid = nn.Sigmoid()
                
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.fc3(x)
                return x
        
        return BinaryClassifier(input_size, self.hidden_size)
    
    def fit(self, X_train: Any, y_train: Any, class_weights: Optional[dict] = None) -> None:
        """Train the PyTorch neural network model with optional class weights."""
        torch, nn, optim, DataLoader, TensorDataset = self._check_pytorch_available()
        
        print("Training simple PyTorch neural network model...")
        
        # Set random seed for reproducibility
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        # Convert data to tensors
        if hasattr(X_train, 'toarray'):  # Handle sparse matrices
            X_train = X_train.toarray()
        
        X_tensor = torch.FloatTensor(X_train)
        y_tensor = torch.LongTensor(y_train.values if hasattr(y_train, 'values') else y_train)
        
        # Store input size and classes
        self.input_size = X_tensor.shape[1]
        self.classes_ = np.unique(y_train)
        
        # Create model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._create_model(self.input_size).to(self.device)
        
        # Move tensors to device
        X_tensor = X_tensor.to(self.device)
        y_tensor = y_tensor.to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Define loss and optimizer with optional class weights
        if class_weights:
            print(f"Using class weights: {class_weights}")
            # Convert class weights to tensor
            weight_list = [class_weights[i] for i in sorted(class_weights.keys())]
            weight_tensor = torch.FloatTensor(weight_list).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            criterion = nn.CrossEntropyLoss()
            
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss/len(dataloader):.4f}')
        
        self.is_fitted = True
        print("Model training completed!")
    
    def predict(self, X: Any) -> np.ndarray:
        """Make predictions using the trained model."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        torch, _, _, _, _ = self._check_pytorch_available()
        
        # Convert data to tensor
        if hasattr(X, 'toarray'):  # Handle sparse matrices
            X = X.toarray()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
        
        return predicted.cpu().numpy()
    
    def predict_proba(self, X: Any) -> np.ndarray:
        """Make probability predictions using the trained model."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        torch, nn, _, _, _ = self._check_pytorch_available()
        
        # Convert data to tensor
        if hasattr(X, 'toarray'):  # Handle sparse matrices
            X = X.toarray()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = nn.functional.softmax(outputs, dim=1)
        
        return probabilities.cpu().numpy()
    
    def get_model_info(self) -> dict:
        """Return information about the PyTorch neural network model."""
        if not self.is_fitted:
            return {"error": "Model not fitted yet"}
        
        return {
            "model_type": "simple_pytorch_neural_network",
            "hidden_size": self.hidden_size,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "random_state": self.random_state,
            "input_size": self.input_size,
            "n_classes": len(self.classes_),
            "classes": self.classes_.tolist(),
            "device": str(self.device)
        } 