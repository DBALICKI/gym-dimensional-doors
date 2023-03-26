import numpy as np


class Obfuscator:
    def __init__(self, difficulty: int):
        self.layers = []
        self.layers.append(ObfuscatorLayer(1, 256))
        self.layers.append(ObfuscatorLayer(256, 1))

    def obfuscate(self, x: float) -> np.ndarray:
        x = np.array([x])
        for layer in self.layers:
            x = layer.forward(x)
        return x.item()


class ObfuscatorLayer:
    def __init__(self, n_inputs: int, n_outputs: int):
        self.weight = np.random.normal(size=(n_inputs, n_outputs))
        self.bias = np.random.normal(size=(n_outputs,))

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weight + self.bias
