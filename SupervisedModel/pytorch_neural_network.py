"""PyTorch neural network model implementation."""

import numpy as np
from typing import Any, Optional
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .base import SupervisedModel


class PyTorchNeuralNetwork(SupervisedModel):
    """PyTorch neural network model for binary classification."""
    
    def __init__(self, hidden_size: int = 128, learning_rate: float = 0.001, 
                 epochs: int = 100, batch_size: int = 32, random_state: int = 42,
                 dropout_rate: float = 0.3, weight_decay: float = 1e-4,
                 patience: int = 10, validation_split: float = 0.2):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.validation_split = validation_split
        self.model = None
        self.device = None
        self.is_fitted = False
        self.input_size = None
        self.classes_ = None
        self.scaler = None
        self.scheduler = None
        
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
        """Create the neural network architecture with improvements."""
        torch, nn, _, _, _ = self._check_pytorch_available()
        
        class ImprovedBinaryClassifier(nn.Module):
            def __init__(self, input_size, hidden_size, dropout_rate):
                super(ImprovedBinaryClassifier, self).__init__()
                
                # More sophisticated architecture with batch normalization
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.bn1 = nn.BatchNorm1d(hidden_size)
                self.dropout1 = nn.Dropout(dropout_rate)
                
                self.fc2 = nn.Linear(hidden_size, hidden_size)
                self.bn2 = nn.BatchNorm1d(hidden_size)
                self.dropout2 = nn.Dropout(dropout_rate)
                
                self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
                self.bn3 = nn.BatchNorm1d(hidden_size // 2)
                self.dropout3 = nn.Dropout(dropout_rate * 0.5)  # Reduce dropout in later layers
                
                self.fc4 = nn.Linear(hidden_size // 2, 2)  # Binary classification
                
                # Use LeakyReLU for better gradient flow
                self.activation = nn.LeakyReLU(0.1)
                
                # Initialize weights properly
                self._initialize_weights()
                
            def _initialize_weights(self):
                """Initialize weights using Xavier/Glorot initialization."""
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
            
            def forward(self, x):
                # First layer
                x = self.fc1(x)
                x = self.bn1(x)
                x = self.activation(x)
                x = self.dropout1(x)
                
                # Second layer with residual connection
                identity = x
                x = self.fc2(x)
                x = self.bn2(x)
                x = self.activation(x)
                x = self.dropout2(x)
                # Add residual connection if dimensions match
                if x.shape == identity.shape:
                    x = x + identity
                
                # Third layer
                x = self.fc3(x)
                x = self.bn3(x)
                x = self.activation(x)
                x = self.dropout3(x)
                
                # Output layer
                x = self.fc4(x)
                return x
        
        return ImprovedBinaryClassifier(input_size, self.hidden_size, self.dropout_rate)
    
    def fit(self, X_train: Any, y_train: Any, class_weights: Optional[dict] = None) -> None:
        """Train the PyTorch neural network model with improved training strategy."""
        torch, nn, optim, DataLoader, TensorDataset = self._check_pytorch_available()
        
        print("Training PyTorch neural network model...")
        
        # Set random seed for reproducibility
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        # Convert data to dense format if sparse
        if hasattr(X_train, 'toarray'):
            X_train = X_train.toarray()
        
        # Normalize features for better training
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Create validation split
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train_scaled, y_train, 
            test_size=self.validation_split, 
            random_state=self.random_state, 
            stratify=y_train
        )
        
        # Convert data to tensors
        X_train_tensor = torch.FloatTensor(X_train_split)
        y_train_tensor = torch.LongTensor(y_train_split.values)
        X_val_tensor = torch.FloatTensor(X_val_split)
        y_val_tensor = torch.LongTensor(y_val_split.values)
        
        # Store input size and classes
        self.input_size = X_train_tensor.shape[1]
        self.classes_ = np.unique(y_train)
        
        # Create model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._create_model(self.input_size).to(self.device)
        
        # Move tensors to device
        X_train_tensor = X_train_tensor.to(self.device)
        y_train_tensor = y_train_tensor.to(self.device)
        X_val_tensor = X_val_tensor.to(self.device)
        y_val_tensor = y_val_tensor.to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Define loss and optimizer with improvements
        if class_weights:
            print(f"Using class weights: {class_weights}")
            weight_list = [class_weights[i] for i in sorted(class_weights.keys())]
            weight_tensor = torch.FloatTensor(weight_list).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            criterion = nn.CrossEntropyLoss()
            
        # Use AdamW optimizer with weight decay
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Training loop with validation and early stopping
        train_losses = []
        val_losses = []
        
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            epoch_train_loss = 0.0
            for batch_X, batch_y in train_dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_dataloader:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    epoch_val_loss += loss.item()
            
            # Calculate average losses
            avg_train_loss = epoch_train_loss / len(train_dataloader)
            avg_val_loss = epoch_val_loss / len(val_dataloader)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(avg_val_loss)
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
            
            # Early stopping
            if patience_counter >= self.patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f'Loaded best model with validation loss: {best_val_loss:.4f}')
        
        self.is_fitted = True
        print("Model training completed!")
    
    def predict(self, X: Any) -> np.ndarray:
        """Make predictions using the trained model."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        torch, _, _, _, _ = self._check_pytorch_available()
        
        # Convert data to dense format if sparse
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        # Apply the same scaling as during training
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
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
        
        # Convert data to dense format if sparse
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        # Apply the same scaling as during training
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
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
            "model_type": "pytorch_neural_network",
            "hidden_size": self.hidden_size,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "random_state": self.random_state,
            "dropout_rate": self.dropout_rate,
            "weight_decay": self.weight_decay,
            "patience": self.patience,
            "validation_split": self.validation_split,
            "input_size": self.input_size,
            "n_classes": len(self.classes_),
            "classes": self.classes_.tolist(),
            "device": str(self.device)
        } 