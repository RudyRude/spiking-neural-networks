# Classifier Examples

This tutorial demonstrates how to use HSNN's classification and regression capabilities for machine learning tasks.

## Table of Contents

- [STDP-Based Classification](#stdp-based-classification)
- [R-STDP Classification](#r-stdp-classification)
- [LSM-Based Classification](#lsm-based-classification)
- [Regression Tasks](#regression-tasks)
- [Performance Evaluation](#performance-evaluation)

## STDP-Based Classification

Unsupervised classification using spike-timing dependent plasticity.

### Basic MNIST Classification

```python
import numpy as np
import lixirnet as ln
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Load MNIST data
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data / 255.0, mnist.target.astype(int)

# Convert to binary images for Poisson encoding
X_binary = (X > 0.5).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_binary, y, test_size=0.2, random_state=42
)

# Create STDP classifier
classifier = ln.STDPClassifier(
    input_size=784,  # 28x28 pixels
    num_classes=10,
    excitatory_neurons=4000,
    inhibitory_neurons=1000,
    learning_rate=0.01,
    homeostatic_rate=0.001
)

# Train the classifier
print("Training STDP classifier...")
classifier.fit(X_train[:1000], y_train[:1000], epochs=10)

# Evaluate
accuracy = classifier.score(X_test[:500], y_test[:500])
print(f"Test accuracy: {accuracy:.3f}")
```

### Custom Encoding

```python
# Rate-based encoding
def rate_encode(image, max_rate=100):
    """Convert pixel intensities to firing rates."""
    return image * max_rate

# Temporal encoding
def temporal_encode(image, duration=0.1, dt=0.001):
    """Convert to temporal spike patterns."""
    spikes = []
    for i, pixel in enumerate(image.flatten()):
        if pixel > 0.1:  # Threshold
            spike_times = np.random.exponential(1/pixel, size=int(duration/dt))
            spike_times = spike_times[spike_times < duration]
            for t in spike_times:
                spikes.append((i, t))
    return sorted(spikes, key=lambda x: x[1])

# Use custom encoding
X_encoded = np.array([rate_encode(img) for img in X_train])
classifier = ln.STDPClassifier(input_size=784, num_classes=10)
classifier.fit(X_encoded, y_train)
```

## R-STDP Classification

Supervised classification with reward-modulated STDP.

### Reward Function Design

```python
def classification_reward(predictions, targets, reward_scale=0.1):
    """Reward function for classification accuracy."""
    correct = (predictions == targets).astype(float)
    return reward_scale * correct

def temporal_reward(predictions, targets, timing_bonus=0.05):
    """Reward with timing bonus for early correct predictions."""
    correct = (predictions == targets)
    timing_factor = np.exp(-np.arange(len(predictions)) * 0.1)
    return timing_bonus * correct * timing_factor
```

### Training with Rewards

```python
# Create R-STDP classifier
classifier = ln.RSTDPClassifier(
    input_size=784,
    hidden_size=2000,
    num_classes=10,
    dopamine_decay=0.9,
    reward_window=0.05,  # 50ms reward integration
    learning_rate=0.01
)

# Custom reward function
def accuracy_reward(network_state, target_class):
    """Reward based on classifier output."""
    output_rates = network_state['output_rates']
    predicted_class = np.argmax(output_rates)
    return 1.0 if predicted_class == target_class else -0.1

classifier.set_reward_function(accuracy_reward)

# Training loop with rewards
for epoch in range(20):
    epoch_accuracy = 0

    for batch_X, batch_y in batch_generator(X_train, y_train, batch_size=32):
        # Simulate and get predictions
        predictions = classifier.predict(batch_X)

        # Calculate rewards
        rewards = np.array([accuracy_reward(pred, true) for pred, true in zip(predictions, batch_y)])

        # Update with rewards
        classifier.update_with_rewards(batch_X, rewards, batch_y)

    # Evaluate
    test_predictions = classifier.predict(X_test[:100])
    accuracy = np.mean(test_predictions == y_test[:100])
    print(f"Epoch {epoch+1}: Accuracy = {accuracy:.3f}")
```

## LSM-Based Classification

Liquid state machine with readout training.

### Basic LSM Classifier

```python
# Create LSM classifier
lsm_classifier = ln.LSMClassifier(
    input_size=784,
    reservoir_size=1000,
    spectral_radius=0.9,  # For Echo State Property
    sparsity=0.1,         # 10% connectivity
    readout_learning_rate=0.01,
    num_classes=10
)

# Initialize reservoir
lsm_classifier.initialize_reservoir(random_seed=42)

# Train readout layer
print("Training readout layer...")
lsm_classifier.fit_readout(X_train[:5000], y_train[:5000])

# Evaluate
train_accuracy = lsm_classifier.score(X_train[:1000], y_train[:1000])
test_accuracy = lsm_classifier.score(X_test[:1000], y_test[:1000])

print(f"Train accuracy: {train_accuracy:.3f}")
print(f"Test accuracy: {test_accuracy:.3f}")
```

### Multi-Reservoir LSM

```python
# Create hierarchical LSM
class MultiReservoirClassifier:
    def __init__(self, input_size, reservoir_sizes, num_classes):
        self.reservoirs = []
        for size in reservoir_sizes:
            reservoir = ln.LSMReservoir(
                input_size=input_size,
                reservoir_size=size,
                spectral_radius=0.95,
                input_scaling=0.5
            )
            self.reservoirs.append(reservoir)

        # Combined readout
        total_features = sum(reservoir_sizes)
        self.readout = ln.LinearReadout(total_features, num_classes)

    def fit(self, X, y):
        # Train each reservoir
        reservoir_outputs = []
        for reservoir in self.reservoirs:
            reservoir.fit(X)  # Adapt reservoir dynamics if needed
            states = reservoir.get_states(X)
            reservoir_outputs.append(states)

        # Concatenate reservoir outputs
        combined_features = np.concatenate(reservoir_outputs, axis=1)

        # Train readout
        self.readout.fit(combined_features, y)

    def predict(self, X):
        reservoir_outputs = []
        for reservoir in self.reservoirs:
            states = reservoir.get_states(X)
            reservoir_outputs.append(states)

        combined_features = np.concatenate(reservoir_outputs, axis=1)
        return self.readout.predict(combined_features)

# Usage
reservoir_sizes = [500, 800, 300]
classifier = MultiReservoirClassifier(784, reservoir_sizes, 10)
classifier.fit(X_train[:2000], y_train[:2000])
accuracy = np.mean(classifier.predict(X_test[:500]) == y_test[:500])
print(f"Multi-reservoir accuracy: {accuracy:.3f}")
```

## Regression Tasks

Using spiking networks for continuous value prediction.

### Time Series Prediction

```python
# Generate synthetic time series
def generate_lorenz_attractor(length=10000, dt=0.01):
    """Generate Lorenz attractor time series."""
    sigma, rho, beta = 10.0, 28.0, 8.0/3.0
    x, y, z = 1.0, 1.0, 1.0
    trajectory = []

    for _ in range(length):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z

        x += dx * dt
        y += dy * dt
        z += dz * dt

        trajectory.append([x, y, z])

    return np.array(trajectory)

# Create Lorenz trajectory
trajectory = generate_lorenz_attractor()

# Create sliding window dataset
window_size = 50
X_reg = []
y_reg = []

for i in range(len(trajectory) - window_size):
    X_reg.append(trajectory[i:i+window_size].flatten())
    y_reg.append(trajectory[i+window_size])  # Predict next point

X_reg = np.array(X_reg)
y_reg = np.array(y_reg)

# Create regression model
regressor = ln.LSMRegressor(
    input_size=window_size * 3,  # 50 points * 3 dimensions
    reservoir_size=2000,
    spectral_radius=0.8,
    readout_learning_rate=0.001
)

# Train
regressor.fit(X_reg[:5000], y_reg[:5000])

# Predict
predictions = regressor.predict(X_reg[5000:5100])
actual = y_reg[5000:5100]

# Calculate MSE
mse = np.mean((predictions - actual)**2)
print(f"Mean squared error: {mse:.6f}")

# Plot prediction
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(actual[:, 0], actual[:, 1], 'b-', alpha=0.7, label='Actual')
plt.plot(predictions[:, 0], predictions[:, 1], 'r--', alpha=0.7, label='Predicted')
plt.title('X-Y Projection')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(actual[:, 0], actual[:, 2], 'b-', alpha=0.7, label='Actual')
plt.plot(predictions[:, 0], predictions[:, 2], 'r--', alpha=0.7, label='Predicted')
plt.title('X-Z Projection')

plt.subplot(1, 3, 3)
plt.plot(actual[:, 1], actual[:, 2], 'b-', alpha=0.7, label='Actual')
plt.plot(predictions[:, 1], predictions[:, 2], 'r--', alpha=0.7, label='Predicted')
plt.title('Y-Z Projection')

plt.tight_layout()
plt.show()
```

### Function Approximation

```python
# Approximate complex function
def target_function(x):
    """Complex function to approximate."""
    return np.sin(2 * np.pi * x) + 0.5 * np.cos(4 * np.pi * x) + 0.3 * np.sin(6 * np.pi * x)

# Generate training data
X_func = np.linspace(0, 2*np.pi, 1000).reshape(-1, 1)
y_func = target_function(X_func.flatten())

# Encode input as spike rates
def encode_input(x, max_rate=50):
    """Encode scalar input as population firing rates."""
    # Simple place coding
    centers = np.linspace(0, 2*np.pi, 100)
    rates = np.exp(-0.5 * ((x - centers) / 0.5)**2) * max_rate
    return rates

X_encoded = np.array([encode_input(x[0]) for x in X_func])

# Create function approximator
function_approx = ln.LSMRegressor(
    input_size=100,  # Population size
    reservoir_size=800,
    spectral_radius=0.9,
    readout_regularization=0.001  # L2 regularization
)

# Train
function_approx.fit(X_encoded, y_func)

# Test
X_test_func = np.linspace(0, 2*np.pi, 200).reshape(-1, 1)
X_test_encoded = np.array([encode_input(x[0]) for x in X_test_func])
predictions = function_approx.predict(X_test_encoded)
actual = target_function(X_test_func.flatten())

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(X_test_func, actual, 'b-', linewidth=2, label='Target Function')
plt.plot(X_test_func, predictions, 'r--', linewidth=2, label='LSM Approximation')
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Function Approximation with LSM')
plt.legend()
plt.grid(True)
plt.show()

# Calculate approximation quality
rmse = np.sqrt(np.mean((predictions - actual)**2))
r_squared = 1 - np.var(predictions - actual) / np.var(actual)
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r_squared:.4f}")
```

## Performance Evaluation

### Classification Metrics

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def evaluate_classifier(classifier, X_test, y_test, class_names=None):
    """Comprehensive classifier evaluation."""
    # Get predictions
    y_pred = classifier.predict(X_test)

    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Additional metrics
    accuracy = np.mean(y_pred == y_test)
    precision = np.mean([np.mean(y_pred[y_test == c] == c) for c in np.unique(y_test)])
    recall = np.mean([np.mean(y_pred == c) for c in np.unique(y_test) if np.sum(y_test == c) > 0])

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }

# Evaluate classifier
if class_names is None:
    class_names = [str(i) for i in range(10)]

metrics = evaluate_classifier(classifier, X_test[:1000], y_test[:1000], class_names)
print(f"Overall Accuracy: {metrics['accuracy']:.3f}")
```

### Regression Metrics

```python
def evaluate_regressor(regressor, X_test, y_test, plot=True):
    """Comprehensive regression evaluation."""
    y_pred = regressor.predict(X_test)

    # Standard metrics
    mse = np.mean((y_pred - y_test)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred - y_test))

    # R-squared
    ss_res = np.sum((y_test - y_pred)**2)
    ss_tot = np.sum((y_test - np.mean(y_test))**2)
    r_squared = 1 - (ss_res / ss_tot)

    print(".4f")
    print(".4f")
    print(".4f")
    print(".4f")

    if plot:
        plt.figure(figsize=(12, 4))

        # Actual vs Predicted
        plt.subplot(1, 3, 1)
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted')

        # Residuals
        residuals = y_test - y_pred
        plt.subplot(1, 3, 2)
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted')
        plt.ylabel('Residual')
        plt.title('Residual Plot')

        # Error distribution
        plt.subplot(1, 3, 3)
        plt.hist(residuals, bins=50, alpha=0.7)
        plt.xlabel('Residual')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')

        plt.tight_layout()
        plt.show()

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r_squared': r_squared
    }

# Evaluate regressor
metrics = evaluate_regressor(regressor, X_test_encoded, y_test_func)