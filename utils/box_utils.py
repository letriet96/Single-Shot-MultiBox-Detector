import torch


def point_form(boxes):
    """ Chuyển default box thành dạng (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) dạng (cx, cy, w, h).
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat(((boxes[:, 2:] + boxes[:, :2])/2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2]), 1)  # w, h


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A, 4].
      box_b: (tensor) bounding boxes, Shape: [B, 4].
    Return:
      (tensor) intersection area, Shape: [A, B].
    """
    A = box_a.size(0)
    B = box_b.size(0)

    # xmax, ymax
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))

    # xmin, ymin
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))

    # xmax - xmin, ymax - ymin -> width, height
    inter = torch.clamp((max_xy - min_xy), min=0)

    return inter[:, :, 0] * inter[:, :, 1]  # width * height


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) ground truth, (xmin, ymin, xmax, ymax), Shape: [num_objects,4]
        box_b: (tensor) default box (xmin, ymin, xmax, ymax), Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A, B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A, B]
    union = area_a + area_b - inter
    return inter / union


def encode(matched, priors, variances):
    """
    Mã hóa thông tin về vị trí của các ground truth so với priors.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form. Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form. Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """
    # Tính toán vector offset (độ lệch) giữa tâm của các ground truth và tâm của các priors
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]

    # Chuẩn hóa, giảm biến thiên của offset
    g_cxcy /= (variances[0] * priors[:, 2:])

    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]

    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors, 4]


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """
    Thực hiện việc gán nhãn cho các priors dựa trên giá trị overlap của chúng với ground truth.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) ground truth (xmin, ymin, xmax, ymax), Shape: [num_obj, num_priors].
        priors: (tensor) định dạng là (cx, cy, w, h), Shape: [n_priors,4].
        variances: (list[float]) Variances of priorboxes
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # mỗi hàng là 1 ground truth, mỗi cột là một priors
    overlaps = jaccard(truths, point_form(priors))  # A, B

    best_prior_overlap, best_prior_idx = overlaps.max(dim=1)  # (A, )

    best_truth_overlap, best_truth_idx = overlaps.max(dim=0)  # (B, )

    # Tại những vị trí best_prior_idx của best_true_overlap thì thay thế bằng 2
    best_truth_overlap.index_fill_(dim=0, index=best_prior_idx, value=2)

    for j in range(best_prior_idx.shape[0]):
        best_truth_idx[best_prior_idx[j]] = j

    matches = truths[best_truth_idx]          # Shape: [num_priors, 4]
    conf = labels[best_truth_idx] + 1         # Shape: [num_priors]
    conf[best_truth_overlap < threshold] = 0  # label as background
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior
