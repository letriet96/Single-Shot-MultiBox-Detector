from lib import *
from make_datapath import make_datapath_list
from dataset import MyDataset, my_collate_fn
from extract_inform_annotation import Anno_xml
from transform import DataTransform
from model import SSD
from multiboxloss import MultiBoxLoss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# prepair data
root_paths = "./data/VOCdevkit/VOC2012"
train_img_lists, train_anno_lists, val_img_lists, val_anno_lists = make_datapath_list(root_paths)

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person",
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

color_mean = (104, 117, 123)
input_size = 300

# dataset
train_dataset = MyDataset(train_img_lists, train_anno_lists, "train", DataTransform(input_size, color_mean),
                          Anno_xml(classes))
val_dataset = MyDataset(val_img_lists, val_anno_lists, "val", DataTransform(input_size, color_mean), Anno_xml(classes))

# dataloader
batch_size = 16
train_loader = data.DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=my_collate_fn)
val_loader = data.DataLoader(val_dataset, batch_size, shuffle=False, collate_fn=my_collate_fn)

dataloader_dict = {"train": train_loader, "val": val_loader}

# Network
cfg = {
    "num_classes": 21,
    "input_size": 300,
    "bbox_aspect_num": [4, 6, 6, 6, 4, 4],  # Trọng số cho các sources từ 1-6
    "feature_map": [38, 19, 10, 5, 3, 1],
    "steps": [8, 16, 32, 64, 100, 300],  # Size of default box
    "min_size": [30, 60, 111, 162, 213, 264],
    "max_size": [60, 111, 162, 213, 264, 315],  #
    "aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2], [2]]  # Tỷ lệ khung hình
}

net = SSD(phase="train", cfg=cfg)
vgg_weights = torch.load("./data/weights/vgg16_reducedfc.pth")
net.vgg.load_state_dict(vgg_weights)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


# He init
net.extras.apply(weights_init)
net.loc.apply(weights_init)
net.conf.apply(weights_init)

# MultiBoxLoss
criterion = MultiBoxLoss(jaccard_threshold=0.5, neg_pos=3, device=device)

# optimizer
optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)


# training, validation
def train_model(net, dataloader_dict, criterion, optimizer, num_epochs):
    # move network to GPU
    net.to(device)

    iteration = 1
    epoch_train_loss = 0.0
    epoch_val_loss = 0.0
    logs = []
    for epoch in range(num_epochs + 1):
        t_epoch_start = time.time()
        t_iter_start = time.time()
        print("---" * 20)
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("---" * 20)
        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
                print("(Training)")
            else:
                if (epoch + 1) % 10 == 0:
                    net.eval()
                    print("---" * 10)
                    print("(Validation)")
                else:
                    continue
            for images, targets in dataloader_dict[phase]:
                # move to GPU
                images = images.to(device)
                targets = [ann.to(device) for ann in targets]
                # init optimizer
                optimizer.zero_grad()
                # forward
                with torch.set_grad_enabled(phase == "train"):
                    outputs = net(images)
                    loss_l, loss_c = criterion(outputs, targets)
                    loss = loss_l + loss_c

                    if phase == "train":
                        loss.backward()  # calculate gradient
                        # Giới hạn giá trị đạo hàm của mạng net ở mức tối đa 2.0
                        nn.utils.clip_grad_value_(net.parameters(), clip_value=2.0)
                        optimizer.step()  # update parameters

                        if (iteration % 10) == 0:
                            t_iter_end = time.time()
                            duration = t_iter_end - t_iter_start
                            print("Iteration {} || Loss: {:.4f} || 10iter: {:.4f} sec".format(iteration, loss.item(),
                                                                                              duration))
                            t_iter_start = time.time()
                        epoch_train_loss += loss.item()
                        iteration += 1
                    else:
                        epoch_val_loss += loss.item()
        t_epoch_end = time.time()
        print("---" * 20)
        print("Epoch {} || epoch_train_loss: {:.4f} || Epoch_val_loss: {:.4f}".format(epoch + 1, epoch_train_loss,
                                                                                      epoch_val_loss))
        print("Duration: {:.4f} sec".format(t_epoch_end - t_epoch_start))
        t_epoch_start = time.time()

        log_epoch = {"epoch": epoch + 1, "train_loss": epoch_train_loss, "val_loss": epoch_val_loss}
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        df.to_csv("./data/ssd_logs.csv")
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0
        if (epoch + 1) % 10 == 0:
            torch.save(net.state_dict(), "./data/weights/ssd300_" + str(epoch + 1) + ".pth")


num_epochs = 1
train_model(net, dataloader_dict, criterion, optimizer, num_epochs=num_epochs)
