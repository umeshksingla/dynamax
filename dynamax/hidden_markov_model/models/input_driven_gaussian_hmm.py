"""
Gaussian hidden Markov model (HMM) with input-driven state transitions.
"""
from typing import NamedTuple, Optional, Tuple, Union
import jax.random as jr
from jaxtyping import Array, Float
import optax

from dynamax.hidden_markov_model.models.abstractions import HMM, HMMParameterSet, HMMPropertySet
from dynamax.hidden_markov_model.models.initial import StandardHMMInitialState, ParamsStandardHMMInitialState
from dynamax.hidden_markov_model.models.transitions import InputDrivenHMMTransitions, ParamsInputDrivenHMMTransitions
from dynamax.hidden_markov_model.models.gaussian_hmm import GaussianHMMEmissions
from dynamax.types import Scalar


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
