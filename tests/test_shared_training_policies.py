"""Analytic policy checks supplement exact saved original training traces."""
import random
import pytest
import numpy as np
import torch
from inr_isr.training.sampling import without_replacement_indices,microbatch_starts
from inr_isr.training.controllers import ramp_weight,update_ema_scalar,clamp_float
from inr_isr.training.state import capture_rng_state,restore_rng_state,atomic_torch_save,atomic_json_save


@pytest.mark.parametrize('size',[-1,0,5,12,15])
def test_global_sampler_keeps_old_indices_and_rng(size):
    """Nonpositive/equal/oversized xyt returns full ordered rows, otherwise randperm."""
    torch.manual_seed(19)
    expected=torch.arange(12) if size<=0 or size>=12 else torch.randperm(12)[:size]
    rng=torch.get_rng_state()
    torch.manual_seed(19)
    actual=without_replacement_indices(12,size,device=torch.device('cpu'))
    assert torch.equal(expected,actual) and torch.equal(rng,torch.get_rng_state())


def test_equal_collocation_still_permutates_with_local_rng():
    """xyzt's equal-count pool permutation must not become xyt's ordered full batch."""
    torch.manual_seed(41);before=torch.get_rng_state()
    expected=torch.randperm(17,generator=torch.Generator(device='cpu').manual_seed(5))
    actual=without_replacement_indices(17,17,generator=torch.Generator(device='cpu').manual_seed(5),
                                       full_when_nonpositive=False,full_when_equal=False)
    assert torch.equal(actual,expected) and torch.equal(before,torch.get_rng_state())


def test_uneven_microbatch_spans_keep_population_weights():
    """Seventeen points become5,5,5,2, not four equally weighted means."""
    assert list(microbatch_starts(17,5))==[0,5,10,15]
    assert [len(range(17)[start:start+5]) for start in microbatch_starts(17,5)]==[5,5,5,2]
    with pytest.raises(ValueError):list(microbatch_starts(17,0))


def test_scalar_rules_retain_first_ema_caps_and_ramp():
    """Known arithmetic protects initialization, bounds and one-based ramp endpoint."""
    assert update_ema_scalar(None,4.,.5)==4.
    assert update_ema_scalar(4.,2.,.5)==3.
    assert update_ema_scalar(float('nan'),2.,.5)==2.
    assert clamp_float(4.,1.,3.)==3.
    with pytest.raises(ValueError):clamp_float(1.,3.,1.)
    assert ramp_weight(2.,1,8,.5)==.5 and ramp_weight(2.,4,8,.5)==2.


def test_rng_roundtrip_restores_python_numpy_and_torch():
    """Checkpoint RNG state restores next draws, not merely seed labels."""
    random.seed(3);np.random.seed(3);torch.manual_seed(3)
    state=capture_rng_state()
    expected=(random.random(),np.random.random(),torch.rand(3))
    restore_rng_state(state)
    actual=(random.random(),np.random.random(),torch.rand(3))
    assert expected[:2]==actual[:2] and torch.equal(expected[2],actual[2])


def test_atomic_state_preserves_payload_and_removes_temporary_names(tmp_path):
    """Both native encodings replace their exact destination and leave no scratch."""
    atomic_torch_save({'value':torch.tensor([1.,2.])},tmp_path/'weights.pt')
    assert torch.equal(torch.load(tmp_path/'weights.pt')['value'],torch.tensor([1.,2.]))
    atomic_json_save({'step':3},tmp_path/'done.json')
    assert __import__('json').loads((tmp_path/'done.json').read_text())=={'step':3}
    assert {p.name for p in tmp_path.iterdir()}=={'weights.pt','done.json'}
