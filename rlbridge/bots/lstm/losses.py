from keras import backend as K
from keras.losses import categorical_crossentropy

__all__ = [
    'policy_loss',
]


def policy_loss(y_true, y_pred):
    entropy = -1 * K.sum(y_pred * K.log(y_pred))
    # Larger entropy == flatter distribution
    # We want to prefer keeping a little bit of entropy, so we subtract
    # it from the loss function.
    return categorical_crossentropy(y_true, y_pred) - 0.02 * entropy
