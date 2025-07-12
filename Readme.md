# 🍎 Hệ thống Phân loại Chất lượng Táo bằng Point Cloud 3D

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

Dự án này là một hệ thống đầu-cuối để tự động phân loại chất lượng táo (Bình thường, Dập, Nứt, Thối) dựa trên việc tái tạo mô hình 3D từ nhiều góc nhìn. Hệ thống sử dụng camera RealSense để thu thập dữ liệu RGB-D, kết hợp YOLO và Segment Anything Model (SAM) để phân đoạn chính xác đối tượng, sau đó hợp nhất các đám mây điểm (point cloud) và cuối cùng sử dụng một mô hình deep learning lai (DGCNN-GAT-Transformer) để phân loại.

---

## 🌟 Tính năng nổi bật

* **Thu thập dữ liệu đa góc nhìn:** Script `capture_multiview.py` giúp dễ dàng chụp ảnh RGB và dữ liệu chiều sâu của một quả táo từ 4 góc độ khác nhau.
* **Phân đoạn thông minh:** Kết hợp **YOLO** để phát hiện ban đầu và **SAM (Segment Anything Model)** với thuật toán đa prompt để tạo ra một mặt nạ (mask) phân đoạn đối tượng cực kỳ chính xác.
* **Tái tạo 3D mạnh mẽ:** Tự động hợp nhất (register) các point cloud từ nhiều góc nhìn bằng thuật toán **ICP (Iterative Closest Point)** và **RANSAC**, tạo ra một mô hình 3D hoàn chỉnh của quả táo.
* **Mô hình phân loại tiên tiến:** Sử dụng kiến trúc lai **DGCNN-GAT-Transformer** để học các đặc trưng hình học từ cục bộ đến toàn cục của đám mây điểm.
* **Pipeline hoàn chỉnh:** Cung cấp một giao diện dòng lệnh (CLI) thân thiện để thực hiện toàn bộ quy trình: Gán nhãn dữ liệu, Huấn luyện mô hình, và Thử nghiệm trên dữ liệu mới.
* **Tăng cường dữ liệu (Data Augmentation):** Tự động xoay, co giãn, và thêm nhiễu (jitter) vào point cloud trong quá trình huấn luyện để tăng độ chính xác và khả năng tổng quát hóa của mô hình.

---

## 📂 Cấu trúc thư mục

```
/3D-Apple-Quality-Classifier/
│
├── data/                 # Chứa dữ liệu thô (ảnh RGB/Depth)
├── dataset/              # Chứa dữ liệu đã xử lý và gán nhãn (.npz)
├── models/               # Chứa các file model đã train (vd: best_dgcnn_model.pt) và checkpoint của SAM
├── results/              # Chứa ảnh kết quả, đồ thị,...
│
├── capture_multiview.py  # Script thu thập dữ liệu
├── apple_quality.py      # Script chính (Gán nhãn, Huấn luyện, Thử nghiệm)
├── requirements.txt      # Các thư viện Python cần thiết
├── .gitignore            # Các file/thư mục bị Git bỏ qua
└── README.md             # File giới thiệu này
```

---

## 🛠️ Cài đặt

1.  **Clone repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/3D-Apple-Quality-Classifier.git](https://github.com/YOUR_USERNAME/3D-Apple-Quality-Classifier.git)
    cd 3D-Apple-Quality-Classifier
    ```

2.  **Tạo và kích hoạt môi trường ảo (khuyến khích):**
    ```bash
    python -m venv venv
    # Trên Windows
    venv\Scripts\activate
    # Trên macOS/Linux
    source venv/bin/activate
    ```

3.  **Cài đặt các thư viện cần thiết:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Tải model checkpoint của SAM:**
    * Tải file checkpoint `sam_vit_b_01ec64.pth` từ [đây](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth).
    * Đặt file vừa tải vào thư mục `models/`.

5.  **(Quan trọng) Cấu hình Roboflow API Key:**
    * Mã nguồn hiện đang chứa API key trực tiếp. Để bảo mật, bạn nên di chuyển nó ra biến môi trường.
    * `RF_API_KEY = "0SwcE0IGoHTNCkcSfoqo"` -> Sửa thành `RF_API_KEY = os.getenv("ROBOFLOW_API_KEY")` và tạo file `.env` hoặc thiết lập biến môi trường hệ thống.

---

## 🚀 Hướng dẫn sử dụng

Hệ thống hoạt động theo một quy trình 4 bước chính.

### Bước 1: Thu thập dữ liệu

Sử dụng camera Intel RealSense để chụp ảnh đa góc nhìn.

```bash
python capture_multiview.py
```
Làm theo hướng dẫn trên màn hình để chụp ảnh cho từng quả táo. Dữ liệu sẽ được lưu vào thư mục `data/`.

### Bước 2: Gán nhãn dữ liệu

Chạy script chính và chọn chức năng `1` để xử lý dữ liệu thô và gán nhãn cho chúng.

```bash
python apple_quality.py
```
* **Chọn `1`** để vào chế độ gán nhãn.
* Cung cấp đường dẫn đến thư mục `data/` (dữ liệu thô).
* Cung cấp đường dẫn đến thư mục `dataset/` (để lưu file đã gán nhãn).
* Hệ thống sẽ xử lý từng quả táo (tái tạo 3D), hiển thị ảnh preview, và hỏi bạn nhãn (0: Normal, 1: Bruised, 2: Cracked, 3: Rotten).
* Các file `.npz` chứa đặc trưng point cloud và nhãn sẽ được lưu vào `dataset/`.

### Bước 3: Huấn luyện mô hình

Sau khi có một bộ dữ liệu đã gán nhãn, bạn có thể bắt đầu huấn luyện.

```bash
python apple_quality.py
```
* **Chọn `2`** để vào chế độ huấn luyện.
* Cung cấp đường dẫn đến thư mục `dataset/`.
* Nhập các tham số huấn luyện như số `epoch`, `batch size`.
* Quá trình huấn luyện sẽ bắt đầu, sử dụng cả tập validation và cơ chế Early Stopping.
* Model có độ chính xác tốt nhất trên tập validation sẽ được lưu lại với tên `best_dgcnn_model.pt`.

### Bước 4: Kiểm tra mô hình

Sử dụng model đã huấn luyện để dự đoán chất lượng của những quả táo mới.

```bash
python apple_quality.py
```
* **Chọn `3`** để vào chế độ kiểm tra.
* Cung cấp đường dẫn đến thư mục chứa dữ liệu mới (có cấu trúc giống thư mục `data/`).
* Hệ thống sẽ xử lý từng quả táo và đưa ra dự đoán về chất lượng cùng với độ tin cậy.

---

## 🔧 Luồng hoạt động chi tiết

1.  **Đầu vào**: 4 cặp ảnh (RGB + Depth) cho mỗi quả táo.
2.  **Phát hiện & Phân đoạn**:
    * Ảnh RGB được đưa vào **YOLO** để tìm bounding box của quả táo.
    * Bounding box và các điểm prompt được tạo ra và đưa vào **SAM** để có được mask chính xác.
3.  **Tái tạo Point Cloud**:
    * Từ ảnh RGB, Depth và mask, một đám mây điểm (point cloud) được tạo ra cho từng góc nhìn.
4.  **Hợp nhất đa góc nhìn**:
    * Các point cloud riêng lẻ được đăng ký (align) với nhau bằng **ICP** và **RANSAC**.
    * Kết quả là một point cloud 3D hoàn chỉnh, dày đặc của quả táo.
5.  **Trích xuất đặc trưng**:
    * Từ point cloud hợp nhất, 10 đặc trưng được trích xuất cho mỗi điểm: `(x, y, z, nx, ny, nz, curvature, r, g, b)`.
6.  **Phân loại**:
    * Dữ liệu đặc trưng được đưa vào mô hình **DGCNN-GAT-Transformer**.
    * **DGCNN** học các đặc trưng hình học cục bộ.
    * **GAT (Graph Attention)** học mối quan hệ giữa các vùng.
    * **Transformer** tổng hợp thông tin toàn cục để đưa ra dự đoán cuối cùng.

---

## ©️ Giấy phép

Dự án này được cấp phép theo Giấy phép MIT. Xem file `LICENSE` để biết thêm chi tiết.