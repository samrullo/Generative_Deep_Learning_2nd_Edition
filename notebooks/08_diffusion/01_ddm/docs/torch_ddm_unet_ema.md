# Updating ema_network parameters
In PyTorch, you can achieve the equivalent of the TensorFlow code using `torch.no_grad()` to prevent tracking the operations for autograd and manually updating the exponential moving average (EMA) of the weights. Here's an example of how you can do it:

```python
import torch

EMA = 0.99

class Model:
    def __init__(self, network, ema_network):
        self.network = network
        self.ema_network = ema_network

    def update_ema(self):
        with torch.no_grad():
            for param, ema_param in zip(self.network.parameters(), self.ema_network.parameters()):
                ema_param.data = EMA * ema_param.data + (1 - EMA) * param.data

# Example usage:
# Define your network and ema_network (they should have the same architecture)
network = ...  # Your model
ema_network = ...  # Your EMA model

model = Model(network, ema_network)
model.update_ema()
```

In this example:

1. `torch.no_grad()` is used to ensure the updates to the EMA weights are not tracked for gradient computation.
2. The `data` attribute of the parameters is used to update the weights directly without creating new tensors, ensuring in-place modification.

Ensure that both `network` and `ema_network` have the same architecture and initial weights (if necessary) before starting the EMA updates.