"""Probability-map reconstruction: known KL values, masks and stable gradients."""
from dataclasses import replace
import numpy as np
import pytest
torch = pytest.importorskip('torch')
from xdiyo_analytics.neural import Reconstruction, HeatmapCNNConfig
from xdiyo_analytics.neural.network import HeatmapEncoder


def uniform_decoder():
    model = HeatmapEncoder(HeatmapCNNConfig(reconstruction=Reconstruction(loss='kl')),2,2,2).double()
    with torch.no_grad():
        for parameter in model.decoder.parameters():
            parameter.zero_()
    return model


def test_kl_target_direction_sum_over_cells_masks_and_count_normalization():
    model = uniform_decoder()
    maps = torch.tensor([[[[.5,0],[.5,0]],[[1.,0],[0,0]]],
                         [[[.25,.25],[.25,.25]],[[0,0],[0,0]]],
                         [[[0.,0],[0,0]],[[0,1.],[0,0]]],
                         [[[0.,0],[0,0]],[[0,0],[0,0]]]],dtype=torch.float64)
    flags = torch.tensor([[1,1],[1,0],[0,1],[0,0]],dtype=torch.float64)
    z = torch.zeros((4,16),dtype=torch.float64,requires_grad=True)
    loss = model.reconstruction_errors(z,maps,flags)
    torch.testing.assert_close(loss,torch.tensor([1.5*np.log(2),0,np.log(4),0],dtype=torch.float64))
    # Convert density/count grids to masses for the loss, independently per map.
    scaled = maps*torch.tensor([7.,1e200],dtype=torch.float64)[None,:,None,None]
    torch.testing.assert_close(loss,model.reconstruction_errors(z,scaled,flags))
    dirty_absent = torch.where(flags[:,:,None,None].bool(),maps,torch.nan)
    torch.testing.assert_close(loss,model.reconstruction_errors(z,dirty_absent,flags))
    loss.sum().backward()
    assert torch.isfinite(z.grad).all()
    assert torch.isfinite(model.decoder[-1].bias.grad).all()
    # Separate softmax per map, not one softmax across both teams.
    with torch.no_grad():
        model.decoder[-1].bias[4:] += 1234
    torch.testing.assert_close(loss,model.reconstruction_errors(z,maps,flags))


def test_extreme_logits_zero_cells_and_finite_backward():
    model = uniform_decoder()
    with torch.no_grad():
        model.decoder[-1].bias[:] = torch.tensor([-10000.,10000.,0,0]*2)
    m = torch.zeros((1,2,2,2),dtype=torch.float64);m[:,:,0,0] = 1
    flags = torch.ones((1,2),dtype=torch.float64)
    z = torch.zeros((1,16),dtype=torch.float64,requires_grad=True)
    loss = model.reconstruction_errors(z,m,flags)
    torch.testing.assert_close(loss,torch.tensor([20000.],dtype=torch.float64))
    loss.sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_kl_positive_target_mass_and_valid_config_required():
    with pytest.raises(ValueError,match='mse or kl'):
        Reconstruction(loss='reverse_kl')
    model = uniform_decoder()
    z = torch.zeros((1,16),dtype=torch.float64)
    for maps,match in [(torch.zeros((1,2,2,2),dtype=torch.float64),'positive mass'),
                       (-torch.ones((1,2,2,2),dtype=torch.float64),'nonnegative'),
                       (torch.full((1,2,2,2),torch.inf,dtype=torch.float64),'finite')]:
        with pytest.raises(ValueError,match=match):
            model.reconstruction_errors(z,maps,torch.ones((1,2),dtype=torch.float64))
    # Valid absence is not a uniform fake target; all-missing is exactly zero.
    value = model.reconstruction_errors(z,torch.zeros((1,2,2,2),dtype=torch.float64),torch.zeros((1,2),dtype=torch.float64))
    assert value.item() == 0


def test_default_mse_values_unchanged_and_kl_encoder_receives_gradients():
    torch.manual_seed(12)
    config = HeatmapCNNConfig(reconstruction=Reconstruction())
    mse = HeatmapEncoder(config,2,4,4)
    maps = torch.rand(2,2,4,4)
    flags = torch.tensor([[1.,0.],[1.,1.]])
    z = mse(maps,flags)
    old = ((mse.decoder(z).reshape_as(maps)-maps).square().mean((2,3))*flags).sum(1)/flags.sum(1).clamp_min(1)
    assert torch.equal(old,mse.reconstruction_errors(z,maps,flags))
    kl = HeatmapEncoder(replace(config,reconstruction=Reconstruction(loss='kl')),2,4,4)
    errors = kl.reconstruction_errors(kl(maps,flags),maps,flags)
    errors.sum().backward()
    for block in (kl.encoder,kl.decoder):
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in block.parameters())
