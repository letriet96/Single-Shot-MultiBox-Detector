import tarfile
import urllib.request
import os


# Khởi tạo biến chứa đường dẫn tới tập dữ liệu
data_dir = "./data"

# Kiểm tra xem thư mục trong data_dir đã tồn tại hay chưa, nếu chưa thì tạo mới thư mục này
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

# Khởi tạo biến url để lưu đường dẫn tới file tập tin dữ liệu cần tải về.
url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'

# Khởi tạo biến target_path để lưu đường dẫn tới tập tin sau khi đã tải về
target_path = os.path.join(data_dir, "VOCtrainval_11-May-2012.tar")

# Kiểm tra xem thư mục trong target_path đã tồn tại trong data_dir hay chưa,
# nếu chưa thì tải về
if not os.path.exists(target_path):
    urllib.request.urlretrieve(url, target_path)

    # Đọc file tar
    tar = tarfile.TarFile(target_path)

    # Giải nén toàn bộ xong để vào data_dir
    tar.extractall(data_dir)
    tar.close()





