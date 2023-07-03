from lib import *


cfg = {
        "num_classes": 21,
        "input_size": 300,
        # "bbox_aspect_num": [4, 6, 6, 6, 4, 4],  # Trọng số cho các sources từ 1-6
        "feature_map": [38, 19, 10, 5, 3, 1],  # Kích thước feature map
        "steps": [8, 16, 32, 64, 100, 300],  # Size of default box
        "min_size": [30, 60, 111, 162, 213, 264],  # Dùng để tính toán thông số dbox
        "max_size": [60, 111, 162, 213, 264, 315],  #
        "aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2], [2]]  # Tỷ lệ khung hình
    }


class DefBox(object):
    """
    Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg):
        super(DefBox, self).__init__()
        self.img_size = cfg["input_size"]
        self.feature_maps = cfg["feature_map"]
        self.min_size = cfg["min_size"]
        self.max_size = cfg["max_size"]
        self.aspect_ratios = cfg["aspect_ratios"]
        self.steps = cfg["steps"]

    def create_defbox(self):
        defbox_list = []

        # Xét từng feature_maps
        for k, f in enumerate(self.feature_maps):
            # Lấy ra tọa độ từng pixel của cái feature_maps đó
            for i, j in itertools.product(range(f), repeat=2):
                # Kích thước của feature map thứ k, sao ko cho bằng f luôn
                f_k = self.img_size / self.steps[k]

                # Tọa độ center point của default box
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # Small box
                s_k = self.min_size[k] / self.img_size
                defbox_list += [cx, cy, s_k, s_k]

                # Big box
                s_k_ = sqrt(s_k * (self.max_size[k] / self.img_size))
                defbox_list += [cx, cy, s_k_, s_k_]

                for ar in self.aspect_ratios[k]:
                    defbox_list += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    defbox_list += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]

        output = torch.Tensor(defbox_list).view(-1, 4)

        # Giới hạn tọa độ của default box
        output.clamp_(min=0, max=1)

        return output


if __name__ == "__main__":
    defbox = DefBox(cfg)
    dbox_list = defbox.create_defbox()
    # print(dbox_list.shape)

    print(dbox_list)

# Tính fk, mà thực ra chả cần tính, nó chính là kích thước của feature map thứ k
# Tính được cx, cy của các dbox ứng với từng pixel theo i, j, fk
# Tính sk theo smax, smin, số feature map
# Tính được w, h theo sk và aspect_ratio[k]
