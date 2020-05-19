import numpy as np
import pytest
import hypothesis.strategies as st
from hypothesis import given, settings

from lstm import LSTM


@settings(deadline=None)
@given(
    st.integers(min_value=1, max_value=10),
    st.integers(min_value=1, max_value=128),
    st.integers(min_value=1, max_value=128),
    st.integers(min_value=1, max_value=128),
    st.booleans(),
    st.floats(min_value=0, max_value=1),
)
def test_lstm_call(
    batch: int,
    seq_len: int,
    input_size: int,
    hidden_size: int,
    bias: bool,
    dropout: float,
) -> None:
    module = LSTM(input_size, hidden_size, 0, bias, dropout)
    x = np.random.rand(seq_len, batch, input_size)
    o = module(x)
