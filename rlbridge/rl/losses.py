from keras import backend as K
from keras.losses import categorical_crossentropy

__all__ = [
    'policy_loss',
]


def policy_loss(y_true, y_pred):
    negative_entropy = K.sum(y_pred * K.log(y_pred))
    return categorical_crossentropy(y_true, y_pred) + 0.1 * negative_entropy
