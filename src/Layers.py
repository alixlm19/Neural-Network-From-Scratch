import numpy as np
from typing import Callable

from . import Activation

class Layer:

    def __init__(self, shape:list[int, int] | int, initial_value: float = 0.0) -> None:

        self._shape = shape

        self._values: np.ndarray = np.zeros(
            shape = self._shape,
            dtype = np.float32
        ) + initial_value

    @property
    def shape(self) -> list[int]:
        return self._shape

class Dense(Layer):

    def __init__(self, units: int, initial_value: float = 0.0, activation: str | Callable = "relu") -> np.ndarray:
        super().__init__(units, initial_value)

        self._weights: np.ndarray | None = None

        self.__activation: Callable = self.set_activation(activation)

    def set_activation(self, activation: str | Callable) -> None:
        if isinstance(activation, str):
            match activation.lower():
                case "relu":
                    self.__activation = Activation.relu
                case "softmax":
                    self.__activation = Activation.softmax
                case "linear":
                    self.__activation = lambda x: x
                case _:
                    raise ValueError(
                        f"Dense layer does not support activation function: {activation}")
        elif callable(activation):
            self.__activation = activation
        else:
            raise ValueError(
                "Argument `activation` must be of type `str` or `function`.")

    def update(self, values: np.ndarray) -> None:
        self._values = self.__activation(values)

    def set_weights(self, shape: list[int, int], initial_value: float = 0.0) -> None:
        self._weights = np.zeros(
            shape,
            dtype = np.float32
        ) + initial_value

    @property
    def info(self):
        
        if not isinstance(self._weights, np.ndarray):
            weight_shape: int  = 0
        else:
            weight_shape: list[int, int] = self._weights.shape

        return f"Dense Layer with {self.shape} units and weight matrix of shape {weight_shape}"