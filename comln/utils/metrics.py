import jax.numpy as jnp


def accuracy(outputs, targets):
    return jnp.mean(jnp.argmax(outputs, axis=-1) == targets)


def accuracy_with_labels(outputs, labels):
    targets = jnp.argmax(labels, axis=-1)
    return accuracy(outputs, targets)
