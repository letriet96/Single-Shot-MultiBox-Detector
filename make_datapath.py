from lib import *


def make_datapath_list(root_path):
    # Mẫu ảnh
    image_path_template = osp.join(root_path, "JPEGImages", "%s.jpg")
    # Mẫu chú thích
    annotation_path_templates = osp.join(root_path, 'Annotations', "%s.xml")

    train_id_names = osp.join(root_path, "ImageSets/Main/train.txt")
    val_id_names = osp.join(root_path, "ImageSets/Main/val.txt")

    train_img_list = []
    train_annotation_list = []
    val_img_list = []
    val_annotation_list = []

    for line in open(train_id_names):
        file_id = line.strip()  # Xóa kí tự xống dòng, khoảng trống
        img_path = (image_path_template % file_id)
        anno_path = (annotation_path_templates % file_id)

        train_img_list.append(img_path)
        train_annotation_list.append(anno_path)

    for line in open(val_id_names):
        file_id = line.strip()
        img_path = (image_path_template % file_id)
        anno_path = (annotation_path_templates % file_id)

        val_img_list.append(img_path)
        val_annotation_list.append(anno_path)

    return train_img_list, train_annotation_list, val_img_list, val_annotation_list


if __name__ == "__main__":
    root_paths = "./data/VOCdevkit/VOC2012"
    train_img_lists, train_annotation_lists, val_img_lists, val_annotation_lists = make_datapath_list(root_paths)
    print(f"Len of train_img_list: {len(train_img_lists)}")
    print(train_img_lists[0])
