from .Layers import Layer

class FFNN:
    """Naive implementation of a feed-forward multi-layer perceptron (neural network)"""
    
    def __init__(self, layers: Layer) -> None:
        self.__layers: list[Layer] = layers
        
        self.initialize_weights()

    def initialize_weights(self):
        for i in range(1, len(self.__layers)):
            prev_layer: Layer = self.__layers[i - 1]
            curr_layer: Layer = self.__layers[i]

            shape = [prev_layer.shape, curr_layer.shape]
            prev_layer.set_weights(shape)

    @property
    def info(self):
        for layer in self.__layers:
            print(layer.info)