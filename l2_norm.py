from lib import *


class L2Norm(nn.Module):
    def __init__(self, input_channels=512, scale=20):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_channels))
        self.scale = scale
        self.reset_parameters()  # gán giá trị self.scale cho toàn bộ phần tử của self.weight
        self.eps = 1e-10

    def reset_parameters(self):
        """
        Trong PyTorch, nn.init.constant_ là một hàm được sử dụng để thiết lập giá trị hằng số
        cho các tham số trong mạng. Hàm này thực hiện việc gán một giá trị nhất định cho một tensor.

        Cú pháp của hàm nn.init.constant_ như sau:
        nn.init.constant_(tensor, value)

        Trong đó:
        tensor: là một tensor PyTorch mà ta muốn thiết lập giá trị hằng số.
        value: là giá trị hằng số được gán cho các phần tử trong tensor.
        Hàm nn.init.constant_ sẽ trực tiếp thay đổi giá trị của tensor truyền vào,
        không tạo ra một tensor mới. Nó được sử dụng chủ yếu trong quá trình khởi tạo
        các tham số mạng với giá trị hằng số.
        """
        nn.init.constant_(self.weight, self.scale)

    def forward(self, x):
        """
        x.size() = batch_size, 512, height, width
        norm.shape = batch_size, 512, 1, 1
        Tóm lại là hàm forward này sẽ lấy từng chiều dữ liệu của x chia cho norm của cái chiều đó

        Norm của từng chiều là căn bậc 2 tổng bình phương các phần tử của chiều dữ liệu đó
        Có tất cả 512 chiều như vậy (dim=1) với mỗi x nên norm tính ra xong sẽ có dạng giống như là
        1 cái vector hàng độ dài 512
        """
        norm = x.pow(2).sum(dim=1, keepdims=True).sqrt() + self.eps
        x = torch.div(x, norm)

        # weight.size = (512) -> (1, 512, 1, 1)
        weight = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)

        # weight là để scale giá trị của x sau khi chuẩn hóa xong
        return weight * x
