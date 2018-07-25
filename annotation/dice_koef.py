
from keras import backend as K

smooth = 1.

def dice_coef_quad(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss_quad(y_true, y_pred):
    return smooth - dice_coef_quad(y_true, y_pred)

def dice_coef_loss(y_true, y_pred):
    return smooth - dice_coef(y_true, y_pred)


def dice_coef_multilabel_quad(y_true, y_pred, numLabels=3):
    dice=0
    for index in range(numLabels):
        dice -= dice_coef_quad(y_true[:,:, index], y_pred[:,:,index])
    return dice

def get_custom_objects():
    custom_objects = {'dice_coef_loss': dice_coef_loss,
                      'dice_coef': dice_coef,
                      'dice_coef_quad':dice_coef_quad,
                      'dice_coef_loss_quad':dice_coef_loss_quad,
                      'dice_coef_multilabel_quad':dice_coef_multilabel_quad
                      }
    return custom_objects