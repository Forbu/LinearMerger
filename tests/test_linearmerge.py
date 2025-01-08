import pytest
import torch
from linearmerge.linearmerge import LinearMerger


@pytest.fixture
def sample_input():
    return torch.randn(32, 10)  # batch_size=32, in_dim=10


def test_linear_merger_initialization():
    # Test upper triangular
    layer = LinearMerger(in_dim=10, out_dim=20, upper=True, random_rows=True)
    assert layer.weight.shape == (20, 10)
    assert layer.bias.shape == (20,)
    assert layer.mask.shape == (20, 10)

    # Test lower triangular
    layer = LinearMerger(in_dim=10, out_dim=20, upper=False, random_rows=True)
    assert layer.weight.shape == (20, 10)
    assert layer.bias.shape == (20,)
    assert layer.mask.shape == (20, 10)


def test_linear_merger_mask_structure():
    # Test upper triangular mask
    layer = LinearMerger(in_dim=5, out_dim=5, upper=True, random_rows=True)
    weight = layer.weight * layer.mask

    # Ensure some elements are zero (due to triangular structure)
    assert torch.sum(layer.mask == 0) > 0

    # Test lower triangular mask
    layer = LinearMerger(in_dim=5, out_dim=5, upper=False, random_rows=True)
    weight = layer.weight * layer.mask

    # Ensure some elements are zero (due to triangular structure)
    assert torch.sum(layer.mask == 0) > 0


def test_linear_merger_forward(sample_input):
    layer = LinearMerger(in_dim=10, out_dim=20, upper=True, random_rows=True)
    output = layer(sample_input)

    # Check output shape
    assert output.shape == (32, 20)

    # Check that output is different from input (transformation happened)
    assert not torch.allclose(output[:, :10], sample_input)


def test_linear_merger_permute_order():
    permute_order = torch.tensor([2, 0, 1])
    layer = LinearMerger(
        in_dim=3, out_dim=3, upper=True, random_rows=False, permute_order=permute_order
    )

    # Check if mask follows the permutation order
    original_mask = torch.triu(torch.ones(3, 3))
    expected_mask = original_mask[permute_order]
    assert torch.allclose(layer.mask, expected_mask)


def test_linear_merger_bias_disabled():
    layer = LinearMerger(
        in_dim=10, out_dim=20, upper=True, random_rows=True, bias=False
    )
    assert not hasattr(layer, "bias")


def test_invalid_initialization():
    with pytest.raises(ValueError):
        # Should raise error when both random_rows and permute_order are False/None
        LinearMerger(in_dim=10, out_dim=20, upper=True, random_rows=False)
