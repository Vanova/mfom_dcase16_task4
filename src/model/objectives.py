"""
# Citations
    "Deep learning with Maximal Figure-of-Merit Cost to Advance Multi-label Speech Attribute Detection"
    Ivan Kukanov, V. Hautam{\"a}ki, Marco Siniscalchi and Kehuang Li.
    "Maximal Figure-of-Merit Embedding for Multi-label Audio Classification"
    I. Kukanov, V. Hautam{\"a}ki, K.A. Lee.
"""
import keras.backend as K

_EPSILON = K.epsilon()


def mfom_eer_normalized(y_true, y_pred):
    """
    NOTE: it is meant to work with 'UvZMisclassification' and 'SmoothErrorCounter' layers.
    Here y_pred is L_k smooth error function
    """
    y_neg = 1 - y_true
    # number of positive samples
    P = K.sum(y_true)
    # number of negative samples
    N = K.sum(y_neg)
    # ===
    # smooth EER
    # ===
    l = y_pred
    fn = l * y_true
    fp = (1. - l) * y_neg
    # === pooled TRUE
    fnr = K.sum(fn) / P
    fpr = K.sum(fp) / N
    smooth_eer = fpr + 4 * K.abs(fnr - fpr)
    return smooth_eer


def mfom_microf1(y_true, y_pred):
    p = 1. - y_pred
    numen = 2. * K.sum(p * y_true)
    denum = K.sum(p + y_true)
    smooth_f1 = numen / denum
    return 1.0 - smooth_f1


def mfom_cprim(y_true, y_pred):
    y_neg = 1 - y_true
    # number of positive samples
    P = K.sum(y_true)
    # number of negative samples
    N = K.sum(y_neg)
    # ===
    # smooth EER
    # ===
    l = y_pred
    fn = l * y_true
    fp = (1. - l) * y_neg
    # === pooled TRUE
    fnr = K.sum(fn) / P
    fpr = K.sum(fp) / N
    smooth_eer = 0.5 * fnr + 10. * fpr
    return smooth_eer


def mfom_eer_embed(y_true, y_pred):
    """
    MFoM embedding: use MFoM scores as new "soft labels", aka Dark Knowledge by Hinton
       y_true: [batch_sz, nclasses]
       y_pred: sigmoid scores, we preprocess these to d_k and l_k, i.e. loss function l_k(Z)
   """
    alpha = 3.
    beta = 0.
    n_embed = 2
    l = _uvz_loss_scores(y_true, y_pred, alpha, beta)
    l_score = 1 - l
    for t in xrange(n_embed):
        l = _uvz_loss_scores(y_true=y_true, y_pred=l_score, alpha=alpha, beta=beta)
        l_score = 1 - l
    # ===
    # MSE(y_pred - l_score), AvgEER = 13.02, NOTE: does not correlate!!!
    # ===
    # mse = K.mean(K.square(y_pred - l_score3), axis=-1)
    # ===
    # binXent(y_pred - l_score), NOTE: higher then baseline
    # ===
    binxent = K.mean(K.binary_crossentropy(y_pred, l_score), axis=-1)  # NOTE: 11.5 EER
    # ===
    # Xent(y_pred - l_score), NOTE: normal func value, EER is not decreasing
    # ===
    # xent = K.categorical_crossentropy(y_pred, l_score)
    return binxent


def _uvz_loss_scores(y_true, y_pred, alpha, beta):
    y_pred = K.clip(y_pred, _EPSILON, 1.0 - _EPSILON)
    y_neg = 1 - y_true
    # Kolmogorov log average of unit labeled models
    unit_avg = y_true * K.exp(y_pred)
    # average over non-zero elements
    unit_avg = K.log(_non_zero_mean(unit_avg))
    # Kolmogorov log average of zero labeled models
    zeros_avg = y_neg * K.exp(y_pred)
    # average over non-zero elements
    zeros_avg = K.log(_non_zero_mean(zeros_avg))
    # misclassification measure, optimized
    d = -y_pred + y_neg * unit_avg + y_true * zeros_avg
    # calculate class loss function l
    l = K.sigmoid(alpha * d + beta)
    return l


def _non_zero_mean(x):
    # All values will meet the criterion > 0
    mask = K.greater(K.abs(x), 0)
    n = K.sum(K.cast(mask, 'float32'), axis=1, keepdims=True)
    return K.sum(x, axis=-1, keepdims=True) / n


MFOM_OBJECTIVES = dict(mfom_eer_normalized=mfom_eer_normalized,
                       mfom_microf1=mfom_microf1,
                       mfom_cprim=mfom_cprim)
