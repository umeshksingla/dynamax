"""
Gaussian hidden Markov model (HMM) with input-driven state transitions.
"""
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from jaxtyping import Array, Float, Int
from tensorflow_probability.substrates import jax as tfp
import tensorflow_probability.substrates.jax.distributions as tfd
import optax

from dynamax.hidden_markov_model.inference import HMMPosterior
from dynamax.hidden_markov_model.models.abstractions import HMM, HMMEmissions, HMMParameterSet, HMMPropertySet, HMMTransitions
from dynamax.hidden_markov_model.models.initial import StandardHMMInitialState, ParamsStandardHMMInitialState
from dynamax.hidden_markov_model.models.transitions import StandardHMMTransitions
# from dynamax.hidden_markov_model.models.linreg_hmm import ParamsLinearRegressionHMMEmissions, LinearRegressionHMMEmissions
from dynamax.hidden_markov_model.models.gaussian_hmm import GaussianHMMEmissions
from dynamax.hidden_markov_model.models.inputdriven_linreg_hmm import InputDrivenHMMTransitions
from dynamax.parameters import ParameterProperties, ParameterSet, PropertySet
from dynamax.types import Scalar, IntScalar
from dynamax.utils.utils import pytree_sum
from dynamax.utils.utils import pytree_slice
from dynamax.utils.bijectors import RealToPSDBijector


class ParamsInputDrivenHMMTransitions(NamedTuple):
    """Parameters for the transitions of an input-driven HMM."""
    weights: Union[Float[Array, "num_states num_states input_dim"], ParameterProperties]
    biases: Union[Float[Array, "num_states num_states"], ParameterProperties]


class ParamsInputDrivenGaussianHMM(NamedTuple):
    """Parameters for a gaussian HMM."""
    initial: ParamsStandardHMMInitialState
    transitions: ParamsInputDrivenHMMTransitions
    emissions: GaussianHMMEmissions


class InputDrivenGaussianHMM(HMM):

    def __init__(self,
                 num_states: int,
                 input_dim: int,
                 emission_dim: int,
                 initial_probs_concentration: Union[Scalar, Float[Array, " num_states"]]=1.1,
                 m_step_optimizer: optax.GradientTransformation = optax.adam(1e-2),
                 m_step_num_iters: int = 50,
                 emission_prior_mean: Union[Scalar, Float[Array, " emission_dim"]] = 0.0,
                 emission_prior_concentration: Scalar = 1e-4,
                 emission_prior_scale: Union[Scalar, Float[Array, "emission_dim emission_dim"]] = 1e-4,
                 emission_prior_extra_df: Scalar = 0.1
                 ):
        self.emission_dim = emission_dim
        self.input_dim = input_dim
        initial_component = StandardHMMInitialState(num_states, initial_probs_concentration=initial_probs_concentration)
        transition_component = InputDrivenHMMTransitions(num_states, input_dim, m_step_optimizer=m_step_optimizer, m_step_num_iters=m_step_num_iters)
        emission_component = GaussianHMMEmissions(num_states, emission_dim,
                                                  emission_prior_mean=emission_prior_mean,
                                                  emission_prior_concentration=emission_prior_concentration,
                                                  emission_prior_scale=emission_prior_scale,
                                                  emission_prior_extra_df=emission_prior_extra_df)
        super().__init__(num_states, initial_component, transition_component, emission_component)

    @property
    def inputs_shape(self):
        """Return the shape of the input."""
        return (self.input_dim,)

    def initialize(self,
                   key: Array=jr.PRNGKey(0),
                   method: str="prior",
                   initial_probs: Optional[Float[Array, " num_states"]]=None,
                   emission_means: Optional[Float[Array, "num_states emission_dim"]]=None,
                   emission_covariances:  Optional[Float[Array, "num_states emission_dim emission_dim"]]=None,
                   emissions:  Optional[Float[Array, "num_timesteps emission_dim"]]=None
        ) -> Tuple[HMMParameterSet, HMMPropertySet]:
        key1, key2, key3 = jr.split(key , 3)
        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(key1, method=method, initial_probs=initial_probs)
        params["transitions"], props["transitions"] = self.transition_component.initialize(key2, method=method)
        params["emissions"], props["emissions"] = self.emission_component.initialize(key3, method=method, emission_means=emission_means, emission_covariances=emission_covariances, emissions=emissions)
        return ParamsInputDrivenGaussianHMM(**params), ParamsInputDrivenGaussianHMM(**props)


# Example usage
if __name__ == "__main__":
    # Setup
    num_states = 3
    input_dim = 5
    emission_dim = 2
    num_timesteps = 10

    key = jr.PRNGKey(0)
    key1, key2, key3 = jr.split(key, 3)

    # Generate random inputs
    inputs = jr.normal(key2, (num_timesteps, input_dim))

    ####################################################
    # # Create transition model
    # transitions_model = InputDrivenHMMTransitions(num_states, input_dim)
    #
    # # Initialize
    # params, props = transitions_model.initialize(key1, method="prior")
    # print("Initialized parameters:")
    # print(f"  weights shape: {params.weights.shape}")
    # print(f"  biases shape: {params.biases.shape}")

    # # Test distribution for a single state and input
    # dist = transitions_model.distribution(params, state=1, inputs=inputs[0])
    # print(f"\nTransition distribution from state 1:")
    # print(dist)
    #
    # # Compute transition matrices
    # trans_matrices = transitions_model._compute_transition_matrices(params, inputs)
    # print(f"\nTransition matrices shape: {trans_matrices.shape}")
    # print(f"First transition matrix (rows should sum to 1):")
    # print(trans_matrices[0])
    # print(f"Row sums: {trans_matrices[0].sum(axis=1)}")
    #
    # print(f"Fourth transition matrix (rows should sum to 1):")
    # print(trans_matrices[3])
    # print(f"Row sums: {trans_matrices[3].sum(axis=1)}")

    ####################################################
    true_idgHMM = InputDrivenGaussianHMM(num_states, input_dim, emission_dim)
    true_params, _ = true_idgHMM.initialize(key1)
    print("Initialized parameters:", true_params)
    true_states, emissions = true_idgHMM.sample(true_params, key2, num_timesteps, inputs)

    # Make a new Gaussian HMM and fit it with EM
    new_idgHMM = InputDrivenGaussianHMM(num_states, input_dim, emission_dim)
    params, props = new_idgHMM.initialize(key3)
    params, lls = new_idgHMM.fit_em(params, props, emissions, inputs, num_iters=10)

    # Plot the marginal log probs across EM iterations
    import matplotlib.pyplot as plt
    plt.plot(lls)
    plt.xlabel("EM iterations")
    plt.ylabel("marginal log prob.")
    plt.show()

    # Use fitted model for posterior inference
    post = new_idgHMM.smoother(params, emissions, inputs)
    print(post.smoothed_probs.shape) # (num_timesteps, num_states)
    ####################################################
