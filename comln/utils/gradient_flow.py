import jax.numpy as jnp

from jax import vmap, grad, nn, tree_util, jit, ops, custom_vjp
from functools import partial
from jax.experimental import ode
from collections import namedtuple


GradientFlowState = namedtuple('GradientFlowState', ['B', 's', 'z'])


def gradient_flow(loss_fn, init_params, inputs, labels, t_final,
                  rtol=1.4e-8, atol=1.4e-8, mxstep=jnp.inf):
    return _gradient_flow(loss_fn, rtol, atol, mxstep, init_params,
                          inputs, labels, t_final)


@partial(custom_vjp, nondiff_argnums=(0, 1, 2, 3))
def _gradient_flow(loss_fn, rtol, atol, mxstep, init_params, inputs, labels, t_final):
    def _dynamics(params, _):
        grads, _ = grad(loss_fn, has_aux=True)(params, inputs, labels)
        return -grads
    trajectory = ode.odeint(
        jit(_dynamics), init_params,
        jnp.asarray([0., t_final], dtype=jnp.float32),
        rtol=rtol, atol=atol, mxstep=mxstep
    )
    return trajectory[-1]

def _gradient_flow_fwd(loss_fn, rtol, atol, mxstep, init_params, inputs, labels, t_final):
    M, N = inputs.shape[0], init_params.shape[0]
    gram = jnp.dot(inputs, inputs.T)
    init_logits = jnp.matmul(inputs, init_params.T)
    diag_indices = jnp.diag_indices(M)
    diag_indices_interlaced = (diag_indices[0], slice(None), diag_indices[1])

    def _dynamics(state, _):
        preds = nn.softmax(init_logits - jnp.matmul(gram, state.s), axis=-1)
        A = (vmap(jnp.diag)(preds) - vmap(jnp.outer)(preds, preds)) / M

        # Update of B
        cross_prod = jnp.einsum('ikn,im,mjnl->ijkl', A, gram, state.B)
        dB = ops.index_add(-cross_prod, diag_indices, A,
            indices_are_sorted=True, unique_indices=True)

        # Update of s
        ds = (preds - labels) / M

        # Update of z
        cross_prod = jnp.einsum('iln,ik,kmjn->imjl', A, gram, state.z)
        As = jnp.einsum('ikl,ml->imk', A, state.s)
        dz = ops.index_add(cross_prod, diag_indices, As,
            indices_are_sorted=True, unique_indices=True)
        dz = ops.index_add(dz, diag_indices_interlaced, As,
            indices_are_sorted=True, unique_indices=True)

        return GradientFlowState(B=dB, s=ds, z=-dz)

    init_state = GradientFlowState(
        B=jnp.zeros((M, M, N, N)),
        s=jnp.zeros((M, N)),
        z=jnp.zeros((M, M, M, N))
    )
    trajectory = ode.odeint(
        jit(_dynamics), init_state,
        jnp.asarray([0., t_final], dtype=jnp.float32),
        rtol=rtol, atol=atol, mxstep=mxstep
    )
    final_state = tree_util.tree_map(lambda x: x[-1], trajectory)
    final_params = init_params - jnp.matmul(final_state.s.T, inputs)
    return final_params, (init_params, inputs, labels, final_state, final_params)

def _gradient_flow_bwd(loss_fn, rtol, atol, mxstep, res, grads_test):
    init_params, inputs, labels, state, params = res
    grads_train, _ = grad(loss_fn, has_aux=True)(params, inputs, labels)

    # Projections
    inputs_grads_test = jnp.matmul(inputs, grads_test.T)
    C = jnp.einsum('ik,ijkl->jl', inputs_grads_test, state.B)
    grads_params = grads_test - jnp.matmul(C.T, inputs)

    D = jnp.einsum('ik,imjk->jm', inputs_grads_test, state.z)
    grads_inputs = -(jnp.matmul(state.s, grads_test)
        + jnp.matmul(C, init_params) + jnp.matmul(D, inputs))

    grads_t_final = -jnp.vdot(grads_train, grads_test)

    return (grads_params, grads_inputs, None, grads_t_final)

_gradient_flow.defvjp(_gradient_flow_fwd, _gradient_flow_bwd)
