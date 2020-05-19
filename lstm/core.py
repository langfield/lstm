""" A pedagogical LSTM module. """
import math
from typing import List, Optional, Tuple

import numpy as np
from asta import Array, symbols, typechecked

N = symbols.N
X = symbols.X
BATCH = symbols.BATCH
SEQ_LEN = symbols.SEQ_LEN
INPUT_SIZE = symbols.INPUT_SIZE
NUM_LAYERS = symbols.NUM_LAYERS
HIDDEN_SIZE = symbols.HIDDEN_SIZE

# pylint: disable=too-few-public-methods, invalid-name


@typechecked
def sigmoid(x: Array[float, -1, ...]) -> Array[float, -1, ...]:
    """ The sigmoid function. """
    return 1 / (1 + np.exp(-x))


@typechecked
def shift_initialization_range(
    x: Array[float, -1, ...], width: float
) -> Array[float, -1, ...]:
    """ Shift from [0,1) initialized values to (-width, width). """
    return (2 * width * x) - width


class LSTMWeights:
    """ A quartet of weight matrices for an LSTM layer. """

    def __init__(self, rows: int, cols: int, init_width: float):
        # Initialize gate weight matrices with values in [0,1).
        self.i: Array[float, rows, cols] = np.random.rand(rows, cols)
        self.f: Array[float, rows, cols] = np.random.rand(rows, cols)
        self.g: Array[float, rows, cols] = np.random.rand(rows, cols)
        self.o: Array[float, rows, cols] = np.random.rand(rows, cols)

        # Compute the weight initialization range and shift to [-\sqrt{k}, \sqrt{k}).
        self.i = shift_initialization_range(self.i, init_width)
        self.f = shift_initialization_range(self.f, init_width)
        self.g = shift_initialization_range(self.g, init_width)
        self.o = shift_initialization_range(self.o, init_width)


class LSTMBiases:
    """ A quartet of bias vectors for an LSTM layer. """

    def __init__(self, size: int, init_width: float):
        # Initialize gate weight matrices with values in [0,1).
        self.i: Array[float, size] = np.random.rand(size,)
        self.f: Array[float, size] = np.random.rand(size,)
        self.g: Array[float, size] = np.random.rand(size,)
        self.o: Array[float, size] = np.random.rand(size,)

        # Compute the weight initialization range and shift to [-\sqrt{k}, \sqrt{k}).
        self.i = shift_initialization_range(self.i, init_width)
        self.f = shift_initialization_range(self.f, init_width)
        self.g = shift_initialization_range(self.g, init_width)
        self.o = shift_initialization_range(self.o, init_width)


@typechecked
class LSTMLayer:
    """ An implementation of an LSTM layer in numpy. """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        layer: int,
        bias: bool = False,
        dropout: float = 0,
    ):
        self.hidden_size = hidden_size

        # Compute initialization width (centered around zero).
        k = 1 / hidden_size
        init_width = math.sqrt(k)

        # Input-hidden weights.
        if layer == 0:
            self.w_i = LSTMWeights(hidden_size, input_size, init_width)
        else:
            self.w_i = LSTMWeights(
                hidden_size, hidden_size, init_width
            )

        # Hidden-hidden weights.
        self.w_h = LSTMWeights(hidden_size, hidden_size, init_width)

        # Input-hidden biases for each layer.
        self.b_i = LSTMBiases(hidden_size, init_width)

        # Hidden-hidden biases for each layer.
        self.b_h = LSTMBiases(hidden_size, init_width)

        # Declarations for hidden and cell states.
        self.h: Array[float, BATCH, HIDDEN_SIZE]
        self.c: Array[float, BATCH, HIDDEN_SIZE]

    def __call__(
        self,
        x: Array[float, SEQ_LEN, BATCH, INPUT_SIZE],
        initial_states: Optional[
            Tuple[
                Array[float, BATCH, HIDDEN_SIZE],
                Array[float, BATCH, HIDDEN_SIZE],
            ]
        ] = None,
    ) -> Tuple[
        Array[float, SEQ_LEN, BATCH, HIDDEN_SIZE],
        Tuple[
            Array[float, BATCH, HIDDEN_SIZE],
            Array[float, BATCH, HIDDEN_SIZE],
        ],
    ]:
        seq_len, batch, input_size = x.shape
        if initial_states:
            self.h, self.c = initial_states[0], initial_states[1]
        else:
            self.h = np.zeros((batch, self.hidden_size))
            self.c = np.zeros((batch, self.hidden_size))

        # The output features ``h_t`` for each ``t``.
        outs = np.zeros((seq_len, batch, self.hidden_size))

        x = x.reshape(batch, seq_len, input_size)
        for b, batch in enumerate(x):
            for t, x_t in enumerate(batch):
                self.h[b], self.c[b] = self._forward(
                    x_t,
                    self.h[b],
                    self.c[b],
                    self.w_i,
                    self.b_i,
                    self.w_h,
                    self.b_h,
                )
                outs[t][b] = self.h[b]

        return outs, (self.h, self.c)

    @staticmethod
    def _gate(
        w_i: Array[float, HIDDEN_SIZE, X],
        x_t: Array[float, X],
        b_i: Array[float, HIDDEN_SIZE],
        w_h: Array[float, HIDDEN_SIZE, HIDDEN_SIZE],
        h: Array[float, HIDDEN_SIZE],
        b_h: Array[float, HIDDEN_SIZE],
    ) -> Array[float, HIDDEN_SIZE]:
        return np.matmul(w_i, x_t) + b_i + np.matmul(w_h, h) + b_h

    def _forward(
        self,
        x_t: Array[float, INPUT_SIZE],
        h_t: Array[float, HIDDEN_SIZE],
        c_t: Array[float, HIDDEN_SIZE],
        w_i: LSTMWeights,
        b_i: LSTMBiases,
        w_h: LSTMWeights,
        b_h: LSTMBiases,
    ) -> Tuple[Array[float, HIDDEN_SIZE], Array[float, HIDDEN_SIZE]]:
        """ The forward function of the module. """
        i_t = sigmoid(self._gate(w_i.i, x_t, b_i.i, w_h.i, h_t, b_h.i))
        f_t = sigmoid(self._gate(w_i.f, x_t, b_i.f, w_h.f, h_t, b_h.f))
        g_t = np.tanh(self._gate(w_i.g, x_t, b_i.g, w_h.g, h_t, b_h.g))
        o_t = sigmoid(self._gate(w_i.o, x_t, b_i.o, w_h.o, h_t, b_h.o))

        c = (f_t * c_t) + (i_t * g_t)
        h = o_t * np.tanh(c)

        return c, h


LSTM = LSTMLayer
