import jax.numpy as jnp
import optax
import math
import json

from jax import jit, nn, random, tree_util, grad
from functools import partial
from collections import namedtuple
from optax import smooth_labels, softmax_cross_entropy
from jax_meta.metalearners.base import MetaLearner, MetaLearnerState

from comln.utils.metrics import accuracy_with_labels
from comln.utils.gradient_flow import gradient_flow


COMLNMetaParameters = namedtuple('COMLNMetaParameters', ['model', 'classifier', 't_final'])


class COMLN(MetaLearner):
    def __init__(
            self,
            model,
            num_ways,
            t_final=1e+0,
            odeint_kwargs='{}',
            eps=0.,
            weight_decay=5e-4
        ):
        super().__init__()
        self.model = model
        self.num_ways = num_ways
        self.t_final = t_final
        self.odeint_kwargs = json.loads(odeint_kwargs)
        self.eps = eps
        self.weight_decay = weight_decay

        self._training = False

    def loss(self, params, inputs, labels):
        logits = jnp.matmul(inputs, params.T)
        loss = jnp.mean(softmax_cross_entropy(logits, labels))
        logs = {
            'loss': loss,
            'accuracy': accuracy_with_labels(logits, labels)
        }
        return loss, ({}, logs)  # No state

    def adapt(self, params, inputs, labels):
        adapted_params = gradient_flow(
            self.loss,
            params.classifier,
            inputs,
            labels,
            jnp.exp(params.t_final),
            **self.odeint_kwargs
        )
        return (adapted_params, {})

    def outer_loss(self, params, state, train, test, args):
        train_features, state = self.model.apply(
            params.model, state, train.inputs, *args
        )
        train_labels = self._smooth_labels(train.targets)
        adapted_params, inner_logs = self.adapt(
            params, train_features, train_labels
        )

        test_features, state = self.model.apply(
            params.model, state, test.inputs, *args
        )
        test_labels = self._smooth_labels(test.targets)
        outer_loss, (_, outer_logs) = self.loss(
            adapted_params, test_features, test_labels
        )
        return (outer_loss, state, inner_logs, outer_logs)

    def meta_init(self, key, *args, **kwargs):
        subkey1, subkey2 = random.split(key)

        model_params, state = self.model.init(subkey1, *args, **kwargs)
        features, _ = self.model.apply(
            model_params, state, *args, **kwargs
        )
        classifier_params = nn.initializers.lecun_normal()(
            subkey2, shape=(self.num_ways, features.shape[-1])
        )

        params = COMLNMetaParameters(
            model=model_params,
            classifier=classifier_params,
            t_final=jnp.asarray(math.log(self.t_final)),
        )
        return (params, state)

    @partial(jit, static_argnums=(0, 5))
    def train_step(self, params, state, train, test, args):
        outer_loss_grad = grad(self.batch_outer_loss, has_aux=True)
        grads, (model_state, logs) = outer_loss_grad(
            params, state.model, train, test, args 
        )

        # Apply weight decay
        grads = grads._replace(
            model=tree_util.tree_map(
                lambda g, p: g + self.weight_decay * p,
                grads.model,
                params.model
            ),
            classifier=grads.classifier + self.weight_decay * params.classifier
        )

        # Apply gradient descent
        updates, opt_state = self.optimizer.update(grads, state.optimizer, params)
        params = optax.apply_updates(params, updates)

        state = MetaLearnerState(model=model_state, optimizer=opt_state)

        return params, state, logs

    def _smooth_labels(self, targets):
        labels = nn.one_hot(targets, num_classes=self.num_ways)
        return smooth_labels(labels, self.eps) if self._training else labels

    def train(self):
        self._training = True

    def eval(self):
        self._training = False
