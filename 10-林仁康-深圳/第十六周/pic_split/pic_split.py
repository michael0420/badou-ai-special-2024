import pixellib
from pixellib.semantic import semantic_segmentation
segment_image = semantic_segmentation()
segment_image.load_pascalvoc_model("model/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels_cityscapes.h5")
segment_image.segmentAsPascalvoc("pic/inout/2.png", output_image_name="pic/output/2_out.png")