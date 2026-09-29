import jax
import jax.numpy as jnp
from bindcraft.loss import interface_residue_pair_loss
from bindcraft.protein import AMINO_ACIDS, ATOM_NAMES, Protein, ResidueFlags, StructurePrediction

def chain(letters, coordinates, flags):
    sequence = jnp.stack([jax.nn.one_hot(AMINO_ACIDS.index(letter), len(AMINO_ACIDS)) for letter in letters])
    atoms = jnp.zeros((len(letters), len(ATOM_NAMES), 3)).at[:, ATOM_NAMES.index('CB')].set(jnp.asarray(coordinates))
    return Protein(sequence=sequence, atoms=atoms, atom_mask=jnp.ones((len(letters), len(ATOM_NAMES)), bool), flags=jnp.asarray(flags, jnp.uint8), residue_index=jnp.arange(len(letters), dtype=jnp.int32))

def binder_target_complex(binder_letters, gap, target_letters):
    design = int(ResidueFlags.DESIGN)
    binder = chain(binder_letters, [[0.0, 0.0, 0.0], [0.0, 0.0, 20.0]], [design, design])
    target = chain(target_letters, [[gap, 0.0, 0.0], [gap, 0.0, 40.0]], [0, 0])
    return {'binder': binder, 'target': target}

def score(binder_letters, gap=6.0, target_letters='EA', **parameters):
    protein_complex = binder_target_complex(binder_letters, gap, target_letters)
    return float(interface_residue_pair_loss({'complex': protein_complex}, {'complex': StructurePrediction(protein_complex, {})}, **parameters))

def test_pair_across_the_interface_is_satisfied():
    assert score('HA') < 0.05

def test_either_direction_counts():
    assert score('EA', target_letters='HA') < 0.05
    assert score('EA', target_letters='HA', symmetric=False) > 0.9
    assert score('HA', target_letters='EA', symmetric=False) < 0.05

def test_missing_or_distant_partner_is_charged():
    assert score('AA') > 0.9
    assert score('HA', gap=14.0) > 0.9

def test_requested_pair_count_saturates():
    assert score('HA', pairs=2.0) > score('HA')
    assert score('HH') <= score('HA') + 1e-06

def test_gradient_points_at_the_contacting_residue():
    protein_complex = binder_target_complex('AA', 6.0, 'EA')

    def logit_loss(logits):
        states = {**protein_complex, 'binder': protein_complex['binder'].replace(sequence=logits)}
        return interface_residue_pair_loss({'complex': states}, {'complex': StructurePrediction(states, {})})

    gradient = jax.grad(logit_loss)(jnp.zeros((2, len(AMINO_ACIDS))))
    assert gradient[0, AMINO_ACIDS.index('H')] < 0
    assert gradient[1, AMINO_ACIDS.index('H')] > -1e-09
