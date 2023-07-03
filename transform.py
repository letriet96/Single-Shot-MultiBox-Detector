from utils.augmentation import Compose, ConvertFromInts, \
                               ToAbsoluteCoords, PhotometricDistort, \
                               Expand, RandomSampleCrop, \
                               RandomMirror, ToPercentCoords, \
                               Resize, SubtractMeans
from extract_inform_annotation import Anno_xml
from lib import cv2, plt
from make_datapath import make_datapath_list


class DataTransform:
    """
    input_size: Là size của ảnh (300)
    color_mean: Lên mạng search mean của tập VOC ra là (104, 117, 123)
    """
    def __init__(self, input_size, color_mean):
        self.data_transform = {
            "train": Compose([ConvertFromInts(),  # Convert image from int to float32
                              ToAbsoluteCoords(),  # Back annotation to normal type
                              PhotometricDistort(),  # Change color by random
                              Expand(color_mean),  #
                              RandomSampleCrop(),  # Cắt ngẫu nhiên một phần ảnh
                              RandomMirror(),  # Xoay ảnh lại theo kiểu phản chiếu qua gương
                              ToPercentCoords(),  # Chuẩn hóa annotation data về dạng 0 1
                              Resize(input_size),  # Resize ảnh về kích thước input_size*inpput_size
                              SubtractMeans(color_mean)  # Resize xong thì trừ mean để chuẩn hóa
                              ]),
            "val": Compose([ConvertFromInts(),
                            Resize(input_size),
                            SubtractMeans(color_mean)
                            ])
        }

    def __call__(self, img, phase, boxes, labels):
        return self.data_transform[phase](img, boxes, labels)


if __name__ == "__main__":
    root_path = "./data/VOCdevkit/VOC2012/"
    train_img_list, train_annotation_list, val_img_list, val_annotation_list = make_datapath_list(root_path)

    idx = 0
    img_file_path = train_img_list[idx]

    img = cv2.imread(img_file_path)  # [height, width, 3 channel BGR]
    height, width, channels = img.shape

    anno_xml = Anno_xml()
    xml_path = train_annotation_list[idx]
    anno_info_list = anno_xml(xml_path, height=height, width=width)

    # Plot original image
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Mặc định cu matplotlib là RGB
    plt.show()

    # Prepair data transform
    color_mean = (104, 117, 123)
    input_size = 300
    transform = DataTransform(input_size, color_mean)

    # Transform train img
    phase = "train"
    img_transformed, boxes, labels = transform(img, phase=phase, boxes=anno_info_list[:, 0:4], labels=anno_info_list[:, 0:4])
    plt.imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB))  # Mặc định cu matplotlib là RGB
    plt.show()

    # Transform val img
    img_transformed, _, __ = transform(img, phase="val", boxes=anno_info_list[:, 0:4],
                                               labels=anno_info_list[:, :4])
    plt.imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB))  # Mặc định cu matplotlib là RGB
    plt.show()
