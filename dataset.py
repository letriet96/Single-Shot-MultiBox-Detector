from lib import *
from extract_inform_annotation import Anno_xml
from make_datapath import make_datapath_list
from transform import DataTransform


class MyDataset(data.Dataset):
    def __init__(self, img_list, anno_list, phase, transform, anno_xml):
        super(MyDataset, self).__init__()
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform
        self.anno_xml = anno_xml

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img, gt = self.pull_item(idx)
        return img, gt

    def pull_item(self, idx):
        img_file_path = self.img_list[idx]
        img = cv2.imread(img_file_path)
        height, width, channel = img.shape

        # Get anno info
        anno_file_path = self.anno_list[idx]
        anno_info = self.anno_xml(anno_file_path, width, height)

        # Preprocessing
        img, boxes, labels = self.transform(img, self.phase, boxes=anno_info[:, :4], labels=anno_info[:, 4])

        # BGR -> RGB and height, width, channel -> channel, height, width
        img = torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1)

        # Ground truth: Chứa cả boxes và label của box đó đã qua transform
        # gt = np.concatenate((boxes, labels.reshape(len(labels), 1)), axis=1)
        gt = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return img, gt


def my_collate_fn(batch):
    """
    Vì mỗi ảnh có thể có nhiều bbox nên cần hàm để sắp xếp dữ liệu cho khớp nhau
    """
    gts = []
    imgs = []
    for sample in batch:  # Sample chính là cái mà Dataset.__getitem__ trả về
        imgs.append(sample[0])  # sample[0] img
        gts.append(torch.FloatTensor(sample[1]))  # sample[1] annotation
    # batch_size, 3, 300, 300  -> Cái này là đã qua transform rồi
    imgs = torch.stack(imgs, dim=0)
    return imgs, gts


if __name__ == "__main__":
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    root_path = "./data/VOCdevkit/VOC2012/"
    train_img_list, train_annotation_list, val_img_list, val_annotation_list = make_datapath_list(root_path)

    color_mean = (104, 117, 123)
    input_size = 300
    train_dataset = MyDataset(img_list=train_img_list, anno_list=train_annotation_list,
                              phase="train", transform=DataTransform(input_size, color_mean),
                              anno_xml=Anno_xml(classes))
    val_dataset = MyDataset(img_list=val_img_list, anno_list=val_annotation_list,
                              phase="val", transform=DataTransform(input_size, color_mean),
                              anno_xml=Anno_xml(classes))

    batch_size = 4
    train_loader = data.DataLoader(train_dataset, batch_size, True, collate_fn=my_collate_fn)
    val_loader = data.DataLoader(val_dataset, batch_size, False, collate_fn=my_collate_fn)

    dataloader_dict = {
        "train": train_loader,
        "val": val_loader
    }
    batch_iter = iter(dataloader_dict["train"])
    images, targets = next(batch_iter)

    print(images.shape)
    print(targets[0])
