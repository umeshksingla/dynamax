"""Module for HMM transition models."""
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Int
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.bijectors as tfb
import optax

from dynamax.hidden_markov_model.models.abstractions import HMMTransitions
from dynamax.hidden_markov_model.inference import HMMPosterior
from dynamax.parameters import ParameterProperties
from dynamax.types import IntScalar, Scalar

from dynamax.utils.utils import pytree_slice
from jaxtyping import Float, Array
from typing import Any, cast, NamedTuple, Optional, Tuple, Union


class ParamsStandardHMMTransitions(NamedTuple):
    """Named tuple for the parameters of the StandardHMMTransitions model."""
    transition_matrix: Union[Float[Array, "state_dim state_dim"], ParameterProperties]


class StandardHMMTransitions(HMMTransitions):
    r"""Standard model for HMM transitions.

    We place a Dirichlet prior over the rows of the transition matrix $A$,

    $$A_k \sim \mathrm{Dir}(\beta 1_K + \kappa e_k)$$

    where

    * $1_K$ denotes a length-$K$ vector of ones,
    * $e_k$ denotes the one-hot vector with a 1 in the $k$-th position,
    * $\beta \in \mathbb{R}_+$ is the concentration, and
    * $\kappa \in \mathbb{R}_+$ is the `stickiness`.

    """
    def __init__(
            self,
            num_states: int,
            concentration: Union[Scalar, Float[Array, "num_states num_states"]]=1.1,
            stickiness: Union[Scalar, Float[Array, " num_states"]]=0.0
    ):
        """
        Args:
            transition_matrix[j,k]: prob(hidden(t) = k | hidden(t-1)j)
        """
        self.num_states = num_states
        self.concentration = \
            concentration * jnp.ones((num_states, num_states)) + \
                stickiness * jnp.eye(num_states)

    def distribution(self, params: ParamsStandardHMMTransitions, state: IntScalar, inputs=None):
        """Return the distribution over the next state given the current state."""
        return tfd.Categorical(probs=params.transition_matrix[state])

    def initialize(
            self,
            key: Optional[Array]=None,
            method="prior",
            transition_matrix: Optional[Float[Array, "num_states num_states"]]=None
    ) -> Tuple[ParamsStandardHMMTransitions, ParamsStandardHMMTransitions]:
        """Initialize the model parameters and their corresponding properties.

        Args:
            key (_type_, optional): _description_. Defaults to None.
            method (str, optional): _description_. Defaults to "prior".
            transition_matrix (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if transition_matrix is None:
            if key is None:
                raise ValueError("key must be provided if transition_matrix is not provided.")
            else:
                transition_matrix_sample = tfd.Dirichlet(self.concentration).sample(seed=key)
                transition_matrix = cast(Float[Array, "num_states num_states"], transition_matrix_sample)

        # Package the results into dictionaries
        params = ParamsStandardHMMTransitions(transition_matrix=transition_matrix)
        props = ParamsStandardHMMTransitions(transition_matrix=ParameterProperties(constrainer=tfb.SoftmaxCentered()))
        return params, props

    def log_prior(self, params: ParamsStandardHMMTransitions) -> Scalar:
        """Compute the log prior probability of the parameters."""
        return tfd.Dirichlet(self.concentration).log_prob(params.transition_matrix).sum()

    def _compute_transition_matrices(
            self, params: ParamsStandardHMMTransitions, inputs=None
    ) -> Float[Array, "num_states num_states"]:
        """Compute the transition matrices."""
        return params.transition_matrix

    def collect_suff_stats(
            self,
            params,
            posterior: HMMPosterior,
            inputs=None
    ) -> Union[Float[Array, "num_states num_states"],
               Float[Array, "num_timesteps_minus_1 num_states num_states"]]:
        """Collect the sufficient statistics for the model."""
        return posterior.trans_probs

    def initialize_m_step_state(self, params, props):
        """Initialize the state for the M-step."""
        return None

    def m_step(
            self,
            params: ParamsStandardHMMTransitions,
            props: ParamsStandardHMMTransitions,
            batch_stats: Float[Array, "batch num_states num_states"],
            m_step_state: Any
        ) -> Tuple[ParamsStandardHMMTransitions, Any]:
        """Perform the M-step of the EM algorithm."""
        if props.transition_matrix.trainable:
            if self.num_states == 1:
                transition_matrix = jnp.array([[1.0]])
            else:
                expected_trans_counts = batch_stats.sum(axis=0)
                transition_matrix = tfd.Dirichlet(self.concentration + expected_trans_counts).mode()
            params = params._replace(transition_matrix=transition_matrix)
        return params, m_step_state


class ParamsInputDrivenHMMTransitions(NamedTuple):
    """Parameters for the transitions of an input-driven HMM."""
    weights: Union[Float[Array, "num_states num_states input_dim"], ParameterProperties]
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
        print("logits.shape", logits.shape)
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
        else:
            raise ValueError(f"Unknown initialization method: {method}")
        # Package the results into dictionaries
        params = ParamsInputDrivenHMMTransitions(weights=weights, biases=biases)
        props = ParamsInputDrivenHMMTransitions(weights=ParameterProperties(), biases=ParameterProperties())
        return params, props

    def log_prior(self, params: ParamsInputDrivenHMMTransitions) -> Scalar:
        """Return the log-prior probability of the emission parameters.

        Currently, there is no prior so this function returns 0.
        """
        return 0.0
