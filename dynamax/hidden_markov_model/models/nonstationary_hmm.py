from typing import NamedTuple, Optional, Tuple, Union
import jax.random as jr
from jaxtyping import Array, Float, Int
import tensorflow_probability.substrates.jax.distributions as tfd
import optax

from dynamax.hidden_markov_model.models.abstractions import HMM, HMMParameterSet, HMMPropertySet, HMMTransitions
from dynamax.hidden_markov_model.models.initial import StandardHMMInitialState, ParamsStandardHMMInitialState
from dynamax.hidden_markov_model.models.gaussian_hmm import GaussianHMMEmissions
from dynamax.parameters import ParameterProperties
from dynamax.types import Scalar


class ParamsInputDrivenHMMTransitions(NamedTuple):
    """Parameters for the transitions of an input-driven HMM."""
    weights: Union[Float[Array, "num_states num_states input_dim"], ParameterProperties]    # CHECK??
    biases: Union[Float[Array, "num_states num_states"], ParameterProperties]


class InputDrivenHMMTransitions(HMMTransitions):
    """
    HMM transitions for an input-driven HMM.
    The transition probabilities depend on external inputs/covariates:
        P(z_t | z_{t-1}, u_t) where u_t are inputs at time t

    For each previous state j, we use multinomial logistic regression:
        P(z_t = k | z_{t-1} = j, u_t) = softmax(W_j @ u_t + b_j)[k]
    """

    def __init__(
            self,
            num_states: int,
            input_dim: int,
            m_step_optimizer: optax.GradientTransformation = optax.adam(1e-2),
            m_step_num_iters: int = 50
    ):
        """
        Args:
            num_states: Number of discrete states
            input_dim: Dimensionality of input vectors
        """
        super().__init__(m_step_optimizer=m_step_optimizer, m_step_num_iters=m_step_num_iters)
        self.num_states = num_states
        self.input_dim = input_dim

    def distribution(
            self,
            params: ParamsInputDrivenHMMTransitions,
            state: Union[int, Int[Array, ""]],
            inputs: Float[Array, " input_dim"]) -> tfd.Distribution:
        """
        Return the distribution over the next state given the current state and input.

        Compute logits: W[state] @ inputs + b[state]
        weights[state] has shape (input_dim, num_states)
        inputs has shape (input_dim,)
        Result has shape (num_states,)
        """
        if inputs is None:
            raise ValueError("Inputs must be provided for input-driven transitions")

        logits = params.weights[state] @ inputs + params.biases[state]
        return tfd.Categorical(logits=logits)

    def initialize(
            self,
            key: Optional[Array] = None,
            method: str = "prior",
            **kwargs
    ) -> Tuple[ParamsInputDrivenHMMTransitions, ParamsInputDrivenHMMTransitions]:
        if method == "prior":
            # Initialize with small random weights (near zero) so transitions start near uniform
            key_w, key_b = jr.split(key)
            weights = jr.normal(key_w, (self.num_states, self.num_states, self.input_dim)) * 0.01
            biases = jr.normal(key_b, (self.num_states, self.num_states)) * 0.01
        else:                                                                               # CHECK??
            raise ValueError(f"Unknown initialization method: {method}")
        # Package the results into dictionaries
        params = ParamsInputDrivenHMMTransitions(weights=weights, biases=biases)
        props = ParamsInputDrivenHMMTransitions(weights=ParameterProperties(), biases=ParameterProperties())
        return params, props

    def log_prior(self, params: ParamsInputDrivenHMMTransitions) -> Scalar:
        return 0.0


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


if __name__ == "__main__":
    # Setup
    num_states = 3
    input_dim = 5
    emission_dim = 2
    num_timesteps = 10

    key = jr.PRNGKey(0)

    # # Generate random inputs
    emissions = jr.normal(key, (num_timesteps, emission_dim))   # (10, 2)
    inputs = jr.normal(key, (num_timesteps, input_dim))         # (10, 5)

    # Create a Gaussian HMM with input-driven state transitions
    hmm = InputDrivenGaussianHMM(num_states, input_dim, emission_dim)
    init_params, props = hmm.initialize(key=key)
    learned_params, lps = hmm.fit_em(init_params, props, emissions=emissions, inputs=inputs, num_iters=5)
