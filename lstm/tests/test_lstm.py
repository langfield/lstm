import numpy as np
import pytest
import hypothesis.strategies as st
from hypothesis import given, settings

from lstm import LSTM


@pytest.mark.skip
@given(
    st.integers(min_value=1, max_value=128),
    st.integers(min_value=1, max_value=128),
    st.integers(min_value=1, max_value=5),
    st.booleans(),
    st.floats(min_value=0, max_value=1),
    st.booleans(),
)
def test_lstm_initialization(
    input_size: int,
    hidden_size: int,
    num_layers: int,
    bias: bool,
    dropout: float,
    bidirectional: bool,
) -> None:
    LSTM(input_size, hidden_size, num_layers, bias, dropout, bidirectional)


@settings(deadline=None)
@given(
    st.integers(min_value=1, max_value=10),
    st.integers(min_value=1, max_value=128),
    st.integers(min_value=1, max_value=128),
    st.integers(min_value=1, max_value=128),
    st.booleans(),
    st.floats(min_value=0, max_value=1),
    st.booleans(),
)
def test_lstm_call(
    batch: int,
    seq_len: int,
    input_size: int,
    hidden_size: int,
    bias: bool,
    dropout: float,
    bidirectional: bool,
) -> None:
    module = LSTM(input_size, hidden_size, 0, bias, dropout, bidirectional)
    x = np.random.rand(seq_len, batch, input_size)
    o = module(x)
