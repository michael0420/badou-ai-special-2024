import os
import numpy as np
from Badou.第十六周8月3日.nets.mrcnn  import get_predict_model
from Badou.第十六周8月3日.nets import resnet,layers,mrcnn_training,mrcnn_training
from Badou.第十六周8月3日.utils.config import Config
from Badou.第十六周8月3日.utils.anchors import get_anchors
from Badou.第十六周8月3日.utils.utils import mold_inputs,unmold_detections
from Badou.第十六周8月3日.utils import visualize
import keras.backend as K

class MASK_RCNN(object):

    _defaults = {
        "model_path": 'model_data/mask_rcnn_coco.h5',
        "classes_path":'model_data/coco_classes.txt',
        "confidence": 0.7,

        #使用coco数据集检测的时候
        "RPN_ANCHOR_SCALES": (32, 64, 128, 256, 512),
        "IMAGE_MIN_DIM": 1024,
        "IMAGE_MAX_DIM": 1024,
    }

    _maks={
        '1' :2,
        '2' :3,
        '46':5

    }


    @classmethod
    def get_defaults(cls,n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n +"'"

    def __init__(self,**kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.config =self._get_config()





    def _get_class(self):
        class_path = os.path.expanduser(self.classes_path)
        with open(class_path) as f:
            classes_names = f.readlines()

        classes_names = [c.strip() for c in classes_names]
        classes_names.insert(0,'BG')
        return classes_names

    def _get_config(self):
        class InferenceConfig(Config):
            NUM_CLASSES = len(self.class_names)
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = self.confidence

            NAME = "shapes"
            RPN_ANCHOR_SCALES = self.RPN_ANCHOR_SCALES
            IMAGE_MIN_DIM = self.IMAGE_MIN_DIM
            IMAGE_MAX_DIM = self.IMAGE_MAX_DIM

        config = InferenceConfig()
        config.display()
        return config


    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model weights must end with .h5'

        # 计算总的种类
        self.num_classes = len(self.class_names)

        self.model = get_predict_model(self.config)
        self.model.load_weights(self.model_path,by_name=True)



    def detect_image(self,image):
        image = [np.array(image)]
        molded_images , image_metas,windows = mold_inputs(self.config,image)

        image_shape = molded_images[0].shape
        anchors = get_anchors(self.config,image_shape)
        anchors = np.broadcast_to(anchors,(1,) + anchors.shape)

        detections, _, _, mrcnn_mask, _, _, _ = \
            self.model.predict([molded_images, image_metas, anchors], verbose=0)

        final_rois , final_class_ids , final_scores , final_masks=\
            unmold_detections(detections[0], mrcnn_mask[0],
                                    image[0].shape, molded_images[0].shape,
                                    windows[0])


        r = {
            'rois': final_rois,
            'class_ids':final_class_ids,
            'scores':final_scores,
            'masks': final_masks
        }

        visualize.display_images(image[0], r['rois'], r['masks'], r['class_ids'],
                                    self.class_names, r['scores'])


    def close_session(self):
        self.sess.close()

