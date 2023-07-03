from lib import *
from model import SSD
from transform import DataTransform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

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

net = SSD(phase="inference", cfg=cfg, nms_thresh=0.4)
net_weights = torch.load("./data/weights/ssd300_mAP_77.43_v2.pth", map_location=device)
net.load_state_dict(net_weights)

img_path = "./data/2e93ba6028b94e7a7d9477370d733820.jpg"

# Preprocessing
img = cv2.imread(img_path)

color_mean = (104, 117, 123)
input_size = 300
transform = DataTransform(input_size, color_mean)

phase = "val"
img_tranformed, boxes, labels = transform(img, phase, "", "")
img_tensor = torch.from_numpy(img_tranformed[:, :, (2, 1, 0)]).permute(2, 0, 1)

# Forward
net.eval()
input = img_tensor.unsqueeze(0)  # (1, 3, 300, 300)
detections = net(input)  # 1, num_class, self.top_k, 5

plt.figure(figsize=(10, 10))
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
font = cv2.FONT_HERSHEY_SIMPLEX

scale = torch.Tensor(img.shape[1::-1]).repeat(2)

for i in range(detections.size(1)):
    j = 0
    while detections[0, i, j, 0] >= 0.6:
        score = detections[0, i, j, 0]
        pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
        cv2.rectangle(img,
                      (int(pt[0]), int(pt[1])),
                      (int(pt[2]), int(pt[3])),
                      colors[i % 3], 2
                      )
        display_text = "%s: %.2f" % (classes[i - 1], score)
        cv2.putText(img, display_text, (int(pt[0]), int(pt[1])),
                    font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        j += 1

cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
