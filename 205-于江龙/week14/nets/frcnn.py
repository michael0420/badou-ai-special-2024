from nets.resnet import ResNet50, classifier_layers
from keras import layers
from nets.RoiPollingConv import RoiPoolingConv
import keras

def get_rpn(base_layers, num_anchors):
    # get region proposal network
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)

    # to get the rpn classification score postive or negative
    x_class = layers.Conv2D(num_anchors, (1, 1), activation="softmax", kernel_initializer='uniform', name='rpn_out_class')(x)
    # get the 4 parameters of the bounding box for each anchor
    x_regr = layers.Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    return [x_class, x_regr, base_layers]

def get_classifier(base_layers, input_rois, num_rois, nb_classes=21, trainable=False):
    pooling_regions = 14
    input_shape = (num_rois, 14, 14, 1024)

    # do the roi pooling based on the input_rois, 
    # make sure the input_rois is the same as the output of the rpn
    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
    # get the output of the classifier
    out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)
    # flatten the output
    out = layers.TimeDistributed(layers.Flatten())(out)

    # get the output of the classifier for each class and the regression
    out_class = layers.TimeDistributed(layers.Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    out_regr = layers.TimeDistributed(layers.Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)

    return [out_class, out_regr]

def get_model(config, num_classes):
    inputs = layers.Input(shape=(None, None, 3))
    roi_input = layers.Input(shape=(None, 4))
    base_layers = ResNet50(inputs)

    num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)
    rpn = get_rpn(base_layers, num_anchors)
    model_rpn = keras.models.Model(inputs, rpn[:2])

    classifier = get_classifier(base_layers, roi_input, config.num_rois, nb_classes=num_classes, trainable=True)
    model_classifier = keras.models.Model([inputs, roi_input], classifier)

    model_all = keras.models.Model([inputs, roi_input], rpn[:2] + classifier)
    return model_rpn, model_classifier, model_all

def get_predict_model(config, num_classes):
    inputs = layers.Input(shape=(None, None, 3))
    roi_input = layers.Input(shape=(None, 4))
    feature_map_input = layers.Input(shape=(None, None, 1024))

    base_layers = ResNet50(inputs)
    num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)
    rpn = get_rpn(base_layers, num_anchors)
    model_rpn = keras.models.Model(inputs, rpn)

    classifier = get_classifier(feature_map_input, roi_input, config.num_rois, nb_classes=num_classes, trainable=True)
    model_classifier_only = keras.models.Model([feature_map_input, roi_input], classifier)

    return model_rpn, model_classifier_only


