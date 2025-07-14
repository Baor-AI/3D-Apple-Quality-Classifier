from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import cv2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import open3d as o3d
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from einops import rearrange
import warnings
import matplotlib.pyplot as plt
from roboflow import Roboflow
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =================== UTILITY FUNCTIONS ===================
def copy_pointcloud(pc):
    """Tạo bản sao của point cloud tương thích với mọi phiên bản Open3D"""
    try:
        # Thử phương thức copy() mới
        return pc.copy()
    except AttributeError:
        # Fallback cho phiên bản cũ
        new_pc = o3d.geometry.PointCloud()
        new_pc.points = o3d.utility.Vector3dVector(np.asarray(pc.points))
        if pc.has_colors():
            new_pc.colors = o3d.utility.Vector3dVector(np.asarray(pc.colors))
        if pc.has_normals():
            new_pc.normals = o3d.utility.Vector3dVector(np.asarray(pc.normals))
        return new_pc
# =================== DATA AUGMENTATION ===================
def random_rotation(points, normals):
    """Xoay ngẫu nhiên đám mây điểm quanh trục Z."""
    theta = np.random.uniform(0, 2 * np.pi)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[cos_t, -sin_t, 0],
                                [sin_t, cos_t, 0],
                                [0, 0, 1]], dtype=np.float32)
    rotated_points = np.dot(points, rotation_matrix)
    rotated_normals = np.dot(normals, rotation_matrix)
    return rotated_points, rotated_normals

def random_jitter(points, sigma=0.01, clip=0.05):
    """Thêm nhiễu ngẫu nhiên (jitter) vào các điểm."""
    B, N, C = points.shape
    assert(C == 3)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += points
    return jittered_data.astype(np.float32)

def random_scaling(points, normals, scale_low=0.8, scale_high=1.2):
    """Co giãn ngẫu nhiên kích thước của đám mây điểm."""
    scale = np.random.uniform(scale_low, scale_high)
    scaled_points = points * scale
    # Normals không bị ảnh hưởng bởi scaling đồng nhất
    return scaled_points.astype(np.float32), normals
# =================== 2. DGCNN + TRANSFORMER MODEL ===================
class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer implementation
    """
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        """
        h: [B, N, in_features]
        adj: [B, N, N] - adjacency matrix
        """
        B, N, _ = h.size()
        
        # Linear transformation
        Wh = torch.matmul(h, self.W)  # [B, N, out_features]
        
        # Compute attention coefficients
        e = self._prepare_attentional_mechanism_input(Wh)  # [B, N, N]
        
        # Mask with adjacency matrix
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # Apply attention to features
        h_prime = torch.matmul(attention, Wh)  # [B, N, out_features]
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        """
        Prepare input for attention mechanism
        """
        B, N, F = Wh.size()
        
        # Create all pairs
        Wh1 = Wh.unsqueeze(2).expand(B, N, N, F)  # [B, N, N, F]
        Wh2 = Wh.unsqueeze(1).expand(B, N, N, F)  # [B, N, N, F]
        
        # Concatenate pairs
        all_combinations = torch.cat([Wh1, Wh2], dim=-1)  # [B, N, N, 2*F]
        
        # Apply attention function
        e = torch.matmul(all_combinations, self.a).squeeze(-1)  # [B, N, N]
        return self.leakyrelu(e)

class MultiHeadGAT(nn.Module):
    """
    Multi-head Graph Attention Network
    """
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.1, alpha=0.2):
        super(MultiHeadGAT, self).__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        
        # Multi-head attention layers
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(in_features, out_features // num_heads, 
                              dropout=dropout, alpha=alpha, concat=True)
            for _ in range(num_heads)
        ])
        
        # Final projection
        self.out_proj = nn.Linear(out_features, out_features)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, adj):
        """
        x: [B, N, in_features]
        adj: [B, N, N]
        """
        # Multi-head attention
        head_outputs = []
        for attention in self.attentions:
            head_out = attention(x, adj)
            head_outputs.append(head_out)
        
        # Concatenate heads
        x = torch.cat(head_outputs, dim=-1)  # [B, N, out_features]
        
        # Final projection
        x = self.out_proj(x)
        x = self.dropout(x)
        
        return x

# =================== GRAPH CONSTRUCTION ===================

def build_knn_graph(points, k=20):
    """
    Build k-NN graph from point cloud
    points: [B, N, 3] - XYZ coordinates
    Returns: [B, N, N] adjacency matrix
    """
    B, N, _ = points.size()
    device = points.device
    
    # Compute pairwise distances
    inner = -2 * torch.matmul(points, points.transpose(-2, -1))
    xx = torch.sum(points ** 2, dim=-1, keepdim=True)
    pairwise_distance = xx + inner + xx.transpose(-2, -1)
    
    # Get k nearest neighbors
    _, idx = torch.topk(-pairwise_distance, k=k, dim=-1)  # [B, N, k]
    
    # Build adjacency matrix
    adj = torch.zeros(B, N, N, device=device)
    for b in range(B):
        for i in range(N):
            adj[b, i, idx[b, i]] = 1.0
    
    # Make symmetric
    adj = torch.maximum(adj, adj.transpose(-2, -1))
    
    return adj

# =================== HYBRID QUERY WITH GAT ===================

def hybrid_knn_ball_query(x, k=20, radius=0.1):
    """
    Fixed hybrid KNN + Ball query
    """
    B, C, N = x.size()
    device = x.device
    
    # Transpose to [B, N, C] for distance computation
    x_t = x.transpose(1, 2)  # [B, N, C]
    
    # Compute pairwise distance (squared)
    inner = -2 * torch.matmul(x_t, x_t.transpose(-2, -1))
    xx = torch.sum(x_t ** 2, dim=-1, keepdim=True)
    pairwise_distance_sq = xx + inner + xx.transpose(-2, -1)
    
    # Convert to actual distances
    pairwise_distance_sq = torch.clamp(pairwise_distance_sq, min=1e-12)
    actual_distances = torch.sqrt(pairwise_distance_sq)
    
    idx_list = []
    
    for b in range(B):
        batch_idx = []
        for n in range(N):
            # Find points in ball
            distances_n = actual_distances[b, n, :]  # [N]
            ball_mask = distances_n <= radius
            ball_indices = torch.where(ball_mask)[0]
            
            if len(ball_indices) >= k:
                # Enough points in ball, take k nearest
                ball_distances = distances_n[ball_indices]
                _, sorted_idx = torch.sort(ball_distances)
                selected_ball_idx = ball_indices[sorted_idx[:k]]
                batch_idx.append(selected_ball_idx)
            else:
                # Not enough points in ball, use KNN
                _, knn_idx = torch.topk(-pairwise_distance_sq[b, n, :], k=k, dim=-1)
                batch_idx.append(knn_idx)
        
        # Stack all indices for this batch
        batch_idx = torch.stack(batch_idx, dim=0)  # [N, k]
        idx_list.append(batch_idx)
    
    idx = torch.stack(idx_list, dim=0)  # [B, N, k]
    return idx

def get_graph_feature(x, k=20, radius=0.1, use_hybrid=True):
    """
    Get graph features for DGCNN with hybrid query
    """
    B, C, N = x.size()
    device = x.device
    
    if use_hybrid:
        idx = hybrid_knn_ball_query(x, k=k, radius=radius)
    else:
        # Original KNN
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        idx = pairwise_distance.topk(k=k, dim=-1)[1]
    
    idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N
    idx = idx + idx_base
    idx = idx.view(-1)
    
    x = x.transpose(2, 1).contiguous()
    feature = x.view(B * N, -1)[idx, :]
    feature = feature.view(B, N, k, C) - x.unsqueeze(2)
    x = x.unsqueeze(2).repeat(1, 1, k, 1)
    feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)
    return feature

class STN3d(nn.Module):
    """Spatial Transformer Network"""
    def __init__(self, input_dim=10):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()

    def forward(self, x):
        B = x.size(0)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        iden = torch.eye(3, device=x.device).view(1, 9).repeat(B, 1)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class EdgeConvBlock(nn.Module):
    """EdgeConv block cho DGCNN"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.mlp(x)
        x = x.max(dim=-1, keepdim=False)[0]
        return x

def geometric_median(x, eps=1e-5, max_iter=100):
    """Tính geometric median cho pooling"""
    z = x.mean(dim=1, keepdim=True)
    for _ in range(max_iter):
        dist = torch.norm(x - z, dim=2, keepdim=True).clamp(min=eps)
        weights = 1.0 / dist
        weights = weights / weights.sum(dim=1, keepdim=True)
        z_new = (weights * x).sum(dim=1, keepdim=True)
        if torch.norm(z - z_new).item() < eps:
            break
        z = z_new
    return z.squeeze(1)
class DGCNN_Transformer(nn.Module):
    """DGCNN + Transformer cho phân loại chất lượng táo với hybrid query"""
    def __init__(self, num_classes, k=20, input_dim=10, ball_radius=0.1, use_hybrid=True,use_gat=True):
        super().__init__()
        self.k = k
        self.input_dim = input_dim
        self.ball_radius = ball_radius
        self.use_hybrid = use_hybrid
        self.use_gat = use_gat
        # STN
        self.stn = STN3d(input_dim=input_dim)

        # DGCNN blocks
        self.conv1 = EdgeConvBlock(input_dim * 2, 64)
        self.conv2 = EdgeConvBlock(64 * 2, 64)
        self.conv3 = EdgeConvBlock(64 * 2, 128)
        self.conv4 = EdgeConvBlock(128 * 2, 256)

        # Transformer
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 128))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=8, batch_first=True),
            num_layers=2
        )
        self.gat1 = MultiHeadGAT(512, 256, num_heads=4)
        self.gat2 = MultiHeadGAT(256, 128, num_heads=4)

        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(128, 256),  # <-- SỬA TỪ 512 THÀNH 128 Ở ĐÂY
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):  # x: [B, N, input_dim]
        B, N, F = x.size()
        x = x.permute(0, 2, 1)  # [B, input_dim, N]

        # Apply STN (chỉ cho XYZ coordinates)
        if self.input_dim >= 3:
            trans = self.stn(x)
            xyz = x[:, :3, :]
            xyz = torch.bmm(trans, xyz)
            if self.input_dim > 3:
                x = torch.cat([xyz, x[:, 3:, :]], dim=1)
            else:
                x = xyz

        # DGCNN layers với hybrid query
        x1 = self.conv1(get_graph_feature(x, k=self.k, radius=self.ball_radius, use_hybrid=self.use_hybrid))
        x2 = self.conv2(get_graph_feature(x1, k=self.k, radius=self.ball_radius, use_hybrid=self.use_hybrid))
        x3 = self.conv3(get_graph_feature(x2, k=self.k, radius=self.ball_radius, use_hybrid=self.use_hybrid))
        x4 = self.conv4(get_graph_feature(x3, k=self.k, radius=self.ball_radius, use_hybrid=self.use_hybrid))

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)  # [B, 512, N]
        x_cat = x_cat.permute(0, 2, 1)  # [B, N, 512]
        points_3d = x[:, :3, :].permute(0, 2, 1)  # [B, N, 3] 
        adj = build_knn_graph(points_3d, k=self.k)
        
        # Apply GAT layers
        x_gat = self.gat1(x_cat, adj)  # [B, N, 256]
        x_gat = self.gat2(x_gat, adj)  # [B, N, 128]
        
        # Continue với Transformer
        cls_token = self.cls_token.expand(B, -1, -1)
        x_gat = torch.cat((cls_token, x_gat), dim=1)
        
        x_trans = self.transformer(x_gat)
        x_cls = x_trans[:, 0, :]
        x_avg = x_trans[:, 1:, :].mean(dim=1)
        x_med = geometric_median(x_trans[:, 1:, :])

        # Fusion
        x_fused = x_cls + x_avg + x_med
        return self.fc(x_fused)

# =================== 3. DATASET CLASS ===================

class AppleDataset(Dataset):
    """Dataset cho training model với data augmentation"""
    def __init__(self, features_list, labels_list, max_points=2048, augment=False): # Thêm augment
        self.features_list = features_list
        self.labels_list = labels_list
        self.max_points = max_points
        self.augment = augment # Lưu trạng thái augment
        
    def __len__(self):
        return len(self.features_list)
    
    def __getitem__(self, idx):
        features = self.features_list[idx]
        label = self.labels_list[idx]
        
        # Sampling/padding to fixed size
        if len(features) > self.max_points:
            indices = np.random.choice(len(features), self.max_points, replace=False)
            features = features[indices]
        elif len(features) < self.max_points:
            n_pad = self.max_points - len(features)
            pad_indices = np.random.choice(len(features), n_pad, replace=True)
            features = np.vstack([features, features[pad_indices]])
            
        # Tách riêng các thành phần đặc trưng
        points = features[:, :3]      # XYZ
        normals = features[:, 3:6]    # Normals
        other_features = features[:, 6:] # Curvature, RGB

        # === ÁP DỤNG AUGMENTATION ===
        if self.augment:
            points, normals = random_rotation(points, normals)
            points = random_jitter(points[np.newaxis, :, :])[0] # Jitter yêu cầu shape (B, N, C)
            points, normals = random_scaling(points, normals)
            
            # Kết hợp lại các đặc trưng sau khi augmentation
            features = np.hstack([points, normals, other_features])

        return torch.FloatTensor(features), torch.LongTensor([label])

# =================== 4. APPLE QUALITY DETECTOR ===================

class AppleQualityDetector:
    """
    Hệ thống phát hiện và phân loại chất lượng táo.
    Đã tích hợp YOLO để phát hiện và SAM để tinh chỉnh vùng chứa đối tượng.
    """
    def __init__(self, 
             # Thêm các tham số cho Roboflow
             rf_api_key: str,
             rf_project: str,
             rf_version: int,
             # Giữ lại các tham số SAM và DGCNN
             sam_checkpoint_path: str = "sam_vit_b_01ec64.pth",
             sam_model_type: str = "vit_b",
             dgcnn_model_path: str = None,
             num_classes: int = 4,
             k: int = 20,
             ball_radius: float = 0.1,
             use_hybrid: bool = True,
             use_gat: bool = True):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

        # --- Tích hợp khởi tạo model từ yolo_detect_tao.txt ---
        logger.info("Đang khởi tạo model YOLO từ Roboflow...")
        try:
            rf = Roboflow(api_key=rf_api_key) 
            project = rf.workspace().project(rf_project) 
            self.yolo_model = project.version(rf_version).model 
            logger.info("✅ Khởi tạo model YOLO từ Roboflow thành công.")
        except Exception as e:
            logger.error(f"❌ Lỗi khi khởi tạo Roboflow: {e}")
            self.yolo_model = None

        # --- Giữ nguyên phần khởi tạo SAM ---
        logger.info(f"Đang tải mô hình SAM ({sam_model_type}) từ {sam_checkpoint_path}...") 
        self.sam_predictor = None
        if Path(sam_checkpoint_path).exists(): 
            try:
                sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint_path) 
                sam.to(device=self.device) 
                self.sam_predictor = SamPredictor(sam) 
                logger.info("✅ Tải mô hình SAM thành công.") 
            except Exception as e:
                logger.error(f"❌ Lỗi khi tải mô hình SAM: {e}") 
        else:
            logger.error(f"❌ Không tìm thấy file checkpoint của SAM: {sam_checkpoint_path}")

        # --- Giữ nguyên phần khởi tạo DGCNN và các thuộc tính khác ---
        self.dgcnn_model = DGCNN_Transformer(
            num_classes=num_classes, input_dim=10, k=k, ball_radius=ball_radius,
            use_hybrid=use_hybrid, use_gat=use_gat
        )
        if dgcnn_model_path and Path(dgcnn_model_path).exists():
            self.dgcnn_model.load_state_dict(torch.load(dgcnn_model_path, map_location='cpu'))
        self.dgcnn_model.eval()

        self.class_names = {0: "Normal", 1: "Bruised", 2: "Cracked", 3: "Rotten"}
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=640, height=480,
            fx=606.8868408203125, fy=606.234375,
            cx=326.2211608886719, cy=250.89784240722656
        )
    def create_pointcloud_from_rgbd(self, rgb_image: np.ndarray, depth_image: np.ndarray, bbox: Tuple[int, int, int, int]) -> o3d.geometry.PointCloud:
        """
        Tạo và trích xuất đám mây điểm của một đối tượng từ ảnh RGB, ảnh chiều sâu và bounding box.
        """
        # Chuyển đổi ảnh numpy thành định dạng của Open3D
        rgb_o3d = o3d.geometry.Image(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
        depth_o3d = o3d.geometry.Image(depth_image)

        # Tạo ảnh RGBD từ ảnh màu và ảnh chiều sâu
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d,
            depth_o3d,
            depth_scale=1000.0,  # Tùy thuộc vào dữ liệu depth của bạn
            depth_trunc=3.0,     # Cắt các giá trị depth quá xa
            convert_rgb_to_intensity=False
        )

        # Tạo đám mây điểm từ ảnh RGBD và thông số camera (intrinsic)
        # self.intrinsic đã được định nghĩa trong __init__ [cite: 51]
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            self.intrinsic
        )

        # Cắt đám mây điểm chỉ lấy vùng trong bounding box
        x1, y1, x2, y2 = bbox
        bbox_o3d = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=(float('-inf'), float('-inf'), float('-inf')),
            max_bound=(float('inf'), float('inf'), float('inf'))
        )
        
        # Tạo bounding box 2D để chọn điểm
        points_2d = np.asarray(pcd.points)
        camera_params = o3d.camera.PinholeCameraParameters()
        camera_params.intrinsic = self.intrinsic
        # Chiếu các điểm 3D ngược lại không gian ảnh 2D để lọc (đây là một cách tiếp cận)
        # Tuy nhiên, cách đơn giản và hiệu quả hơn là tạo một bounding box trong không gian 3D
        # bằng cách sử dụng các điểm đã biết.
        # Ở đây, ta sẽ dùng một phương pháp đơn giản hơn: crop trực tiếp trên ảnh
        
        # Tạo một bounding box 3D để crop
        cropping_box = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=(x1, y1, -1),
            max_bound=(x2, y2, 1)
        )
        # Phương pháp trên không chính xác vì bbox là 2D. 
        # Cách đúng là tạo point cloud rồi lọc.
        
        # Chọn các điểm nằm trong bounding box 2D
        # Đây là một cách đơn giản hóa để lọc các điểm
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        # Chúng ta cần một ma trận biến đổi để chiếu điểm 3D về 2D,
        # nhưng một cách đơn giản hơn là giả định các chỉ số của pcd tương ứng với pixel.
        indices_to_keep = []
        # Điều này chỉ đúng nếu point cloud không bị biến đổi và có cùng kích thước với ảnh
        # Cách tiếp cận đúng là duyệt qua các pixel trong bbox và tìm điểm 3D tương ứng.
        
        # Cách đơn giản nhất và hiệu quả nhất là tạo point cloud chỉ từ vùng ảnh đã crop
        cropped_rgb = rgb_image[y1:y2, x1:x2].copy()
        cropped_depth = depth_image[y1:y2, x1:x2].copy()
        
        # Cần điều chỉnh lại intrinsic cho ảnh đã crop
        cropped_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=cropped_rgb.shape[1],
            height=cropped_rgb.shape[0],
            fx=self.intrinsic.get_focal_length()[0],
            fy=self.intrinsic.get_focal_length()[1],
            cx=self.intrinsic.get_principal_point()[0] - x1,
            cy=self.intrinsic.get_principal_point()[1] - y1
        )
        
        cropped_rgb_o3d = o3d.geometry.Image(cv2.cvtColor(cropped_rgb, cv2.COLOR_BGR2RGB))
        cropped_depth_o3d = o3d.geometry.Image(cropped_depth)
        
        cropped_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            cropped_rgb_o3d, cropped_depth_o3d, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False
        )
        
        final_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            cropped_rgbd, cropped_intrinsic
        )

        return final_pcd
    def create_enhanced_prompts(self, detection, image_shape):
        """Tạo nhiều prompt nâng cao với focus vào hình dạng tròn"""
        x, y, w, h = detection["x"], detection["y"], detection["width"], detection["height"]
        h_img, w_img = image_shape[:2]
        
        prompts = []
        
        # 1. Box gốc
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)
        prompts.append(("original_box", np.array([[x1, y1, x2, y2]])))
        
        # 2. Box vuông (để khuyến khích hình dạng tròn hơn)
        size = min(w, h)
        x1_sq = max(0, int(x - size/2))
        y1_sq = max(0, int(y - size/2))
        x2_sq = min(w_img, int(x + size/2))
        y2_sq = min(h_img, int(y + size/2))
        prompts.append(("square_box", np.array([[x1_sq, y1_sq, x2_sq, y2_sq]])))
        
        # 3. Box vuông lớn hơn
        size_large = max(w, h)
        x1_sql = max(0, int(x - size_large/2))
        y1_sql = max(0, int(y - size_large/2))
        x2_sql = min(w_img, int(x + size_large/2))
        y2_sql = min(h_img, int(y + size_large/2))
        prompts.append(("large_square_box", np.array([[x1_sql, y1_sql, x2_sql, y2_sql]])))
        
        # 4. Box mở rộng với tỷ lệ khác nhau
        for ratio, name in [(0.1, "small_expand"), (0.2, "medium_expand"), (0.35, "large_expand")]:
            w_exp, h_exp = w * (1 + ratio), h * (1 + ratio)
            x1_exp = max(0, int(x - w_exp/2))
            y1_exp = max(0, int(y - h_exp/2))
            x2_exp = min(w_img, int(x + w_exp/2))
            y2_exp = min(h_img, int(y + h_exp/2))
            prompts.append((f"{name}_box", np.array([[x1_exp, y1_exp, x2_exp, y2_exp]])))
        
        # 5. Point prompts - Center
        center_point = np.array([[int(x), int(y)]])
        center_label = np.array([1])
        prompts.append(("center_point", center_point, center_label))
        
        # 6. Multiple points - Grid pattern
        grid_points = []
        grid_labels = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                px = int(x + i * w/6)
                py = int(y + j * h/6)
                if 0 <= px < w_img and 0 <= py < h_img:
                    grid_points.append([px, py])
                    grid_labels.append(1)
        
        if grid_points:
            prompts.append(("grid_points", np.array(grid_points), np.array(grid_labels)))
        
        # 7. Circular pattern points
        radius = min(w, h) / 4
        circular_points = []
        circular_labels = []
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            px = int(x + radius * np.cos(angle))
            py = int(y + radius * np.sin(angle))
            if 0 <= px < w_img and 0 <= py < h_img:
                circular_points.append([px, py])
                circular_labels.append(1)
        
        # Add center point
        circular_points.append([int(x), int(y)])
        circular_labels.append(1)
        
        if circular_points:
            prompts.append(("circular_points", np.array(circular_points), np.array(circular_labels)))
        
        return prompts
    
    def enhanced_mask_evaluation(self, mask, detection, image_shape):
        """Đánh giá mask cải tiến với trọng số cho hình dạng tròn"""
        x, y, w, h = detection["x"], detection["y"], detection["width"], detection["height"]
        
        # 1. Coverage của bounding box
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)
        box_mask = np.zeros_like(mask, dtype=np.uint8)
        box_mask[y1:y2, x1:x2] = 1
        
        intersection = np.logical_and(mask, box_mask).sum()
        union = np.logical_or(mask, box_mask).sum()
        iou = intersection / union if union > 0 else 0
        
        # 2. Tỷ lệ diện tích
        mask_area = mask.sum()
        box_area = w * h
        area_ratio = mask_area / box_area if box_area > 0 else 0
        
        # 3. Compactness nâng cao
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(largest_contour, True)
            if perimeter > 0:
                compactness = 4 * np.pi * mask_area / (perimeter ** 2)
            else:
                compactness = 0
            
            # 4. Aspect ratio của mask
            mask_bbox = cv2.boundingRect(largest_contour)
            mask_width, mask_height = mask_bbox[2], mask_bbox[3]
            mask_aspect = max(mask_width, mask_height) / min(mask_width, mask_height) if min(mask_width, mask_height) > 0 else 1
            
            # 5. Solidity (độ đặc)
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            solidity = mask_area / hull_area if hull_area > 0 else 0
            
            # 6. Extent (tỷ lệ mask trong bounding box)
            extent = mask_area / (mask_width * mask_height) if (mask_width * mask_height) > 0 else 0
            
        else:
            compactness = 0
            mask_aspect = 1
            solidity = 0
            extent = 0
        
        # 7. Tỷ lệ mask trong box (coverage)
        mask_in_box = mask[y1:y2, x1:x2].sum()
        box_coverage = mask_in_box / box_area if box_area > 0 else 0
        
        # 8. Penalty cho hình dạng quá kéo dài
        shape_penalty = 1.0
        if mask_aspect > 1.5:  # Quá kéo dài
            shape_penalty = 1.0 / (1.0 + (mask_aspect - 1.5) * 0.5)
        
        # 9. Penalty cho diện tích quá nhỏ hoặc quá lớn
        min_area_ratio = 0.15  # Mask phải chiếm ít nhất 15% diện tích box
        max_area_ratio = 2.0   # Mask không được quá 200% diện tích box
        
        size_penalty = 1.0
        if area_ratio < min_area_ratio:
            size_penalty = area_ratio / min_area_ratio
        elif area_ratio > max_area_ratio:
            size_penalty = max_area_ratio / area_ratio
        
        # 10. Bonus cho hình dạng tròn
        circularity_bonus = 1.0
        if compactness > 0.7:  # Hình dạng tròn tốt
            circularity_bonus = 1.0 + (compactness - 0.7) * 0.3
        
        # Tính score tổng hợp với trọng số cải tiến
        score = (
            iou * 0.25 + 
            box_coverage * 0.25 + 
            compactness * 0.30 +  # Tăng trọng số cho compactness
            solidity * 0.10 +
            extent * 0.10
        ) * shape_penalty * size_penalty * circularity_bonus
        
        return {
            'score': score,
            'iou': iou,
            'area_ratio': area_ratio,
            'compactness': compactness,
            'box_coverage': box_coverage,
            'mask_area': mask_area,
            'mask_aspect': mask_aspect,
            'solidity': solidity,
            'extent': extent,
            'shape_penalty': shape_penalty,
            'size_penalty': size_penalty,
            'circularity_bonus': circularity_bonus
        }
    
    def segment_with_sam(self, image, prompts):
        """Segment với SAM sử dụng nhiều prompt"""
        print("🔍 Segmenting with SAM...")
        self.sam_predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        all_results = []
        
        for prompt_data in prompts:
            prompt_name = prompt_data[0]
            
            try:
                if "point" in prompt_name:
                    # Point prompts
                    point_coords = prompt_data[1]
                    point_labels = prompt_data[2]
                    masks, scores, _ = self.sam_predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        multimask_output=True
                    )
                else:
                    # Box prompts
                    box = prompt_data[1]
                    masks, scores, _ = self.sam_predictor.predict(
                        box=box,
                        multimask_output=True
                    )
                
                # Lưu tất cả masks từ prompt này
                for i, (mask, score) in enumerate(zip(masks, scores)):
                    all_results.append({
                        'mask': mask,
                        'sam_score': score,
                        'prompt_name': f"{prompt_name}_{i}",
                        'prompt_type': prompt_name
                    })
                    
            except Exception as e:
                print(f"⚠️ Lỗi với prompt {prompt_name}: {e}")
                continue
        
        
        return all_results
    
    def find_best_mask_enhanced(self, sam_results, detection, image_shape):
        """Tìm mask tốt nhất với thuật toán cải tiến"""
        best_mask = None
        best_score = -1
        best_info = None
        
        candidates = []
        
        for result in sam_results:
            mask = result['mask']
            evaluation = self.enhanced_mask_evaluation(mask, detection, image_shape)
            
            # Kết hợp SAM score và evaluation score
            combined_score = 0.4 * evaluation['score'] + 0.6 * result['sam_score']
            
            candidates.append({
                'mask': mask,
                'evaluation': evaluation,
                'combined_score': combined_score,
                'prompt_name': result['prompt_name'],
                'sam_score': result['sam_score']
            })
        
        # Sắp xếp theo combined score
        candidates.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Lọc thêm dựa trên các tiêu chí nghiêm ngặt
        filtered_candidates = []
        for candidate in candidates:
            eval_info = candidate['evaluation']
            
            # Bỏ qua những mask quá kéo dài
            if eval_info['mask_aspect'] > 2.0:
                continue
            
            # Bỏ qua những mask quá nhỏ hoặc quá lớn
            if eval_info['area_ratio'] < 0.1 or eval_info['area_ratio'] > 3.0:
                continue
            
            # Bỏ qua những mask có compactness quá thấp
            if eval_info['compactness'] < 0.3:
                continue
            
            filtered_candidates.append(candidate)
        
        if not filtered_candidates:
            print("⚠️ Không có candidate nào pass được bộ lọc nghiêm ngặt, dùng candidate tốt nhất")
            filtered_candidates = candidates[:1]
        
        
        # Chọn candidate tốt nhất
        best_candidate = filtered_candidates[0]
        best_mask = best_candidate['mask']
        best_info = {
            'prompt_name': best_candidate['prompt_name'],
            'sam_score': best_candidate['sam_score'],
            'evaluation': best_candidate['evaluation'],
            'combined_score': best_candidate['combined_score']
        }
        
        return best_mask, best_info
    
    def create_adaptive_final_contour(self, mask, detection, adaptive_scale=True):
        """Tạo final contour với scaling thích ứng"""
        # Tính toán scale factor thích ứng
        if adaptive_scale:
            # Phân tích hình dạng để quyết định scale factor
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                
                if perimeter > 0:
                    compactness = 4 * np.pi * area / (perimeter ** 2)
                    
                    # Scale factor thích ứng dựa trên compactness
                    if compactness > 0.8:  # Rất tròn
                        scale_factor = 1.05
                    elif compactness > 0.6:  # Tròn vừa
                        scale_factor = 1.1
                    elif compactness > 0.4:  # Hơi kéo dài
                        scale_factor = 1.15
                    else:  # Kéo dài nhiều
                        scale_factor = 1.2
                else:
                    scale_factor = 1.15
            else:
                scale_factor = 1.15
        else:
            scale_factor = 1.15
        
        
        # Làm mịn mask với morphological operations nâng cao
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Adaptive morphological operations
        kernel_size = max(3, int(min(detection["width"], detection["height"]) / 30))
        kernel_close = np.ones((kernel_size, kernel_size), np.uint8)
        kernel_open = np.ones((max(3, kernel_size-2), max(3, kernel_size-2)), np.uint8)
        
        # Đóng lỗ hổng
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel_close)
        # Loại bỏ nhiễu
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel_open)
        
        # Gaussian blur thích ứng
        blur_size = max(3, kernel_size // 2)
        if blur_size % 2 == 0:
            blur_size += 1
        mask_uint8 = cv2.GaussianBlur(mask_uint8, (blur_size, blur_size), 1)
        
        # Threshold lại
        _, mask_uint8 = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)
        
        # Tìm contours
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None, None
        
        # Lấy contour lớn nhất
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Làm mịn contour với epsilon thích ứng
        perimeter = cv2.arcLength(largest_contour, True)
        epsilon = 0.002 * perimeter  # Giảm epsilon để giữ chi tiết
        smooth_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Tính tâm khối lượng
        M = cv2.moments(smooth_contour)
        if M["m00"] == 0:
            return None, None, None
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Tạo final contour với scaling thông minh
        contour_points = smooth_contour.squeeze().astype(np.float32)
        if len(contour_points.shape) == 1:
            contour_points = contour_points.reshape(1, -1)
        
        # Scale từ tâm khối lượng với giới hạn
        centered_points = contour_points - [cx, cy]
        scaled_points = centered_points * scale_factor
        final_contour = (scaled_points + [cx, cy]).astype(np.int32)
        
        # Tạo mask cuối cùng từ final contour
        final_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.fillPoly(final_mask, [final_contour], 255)
        
        return smooth_contour, final_contour, final_mask
    
    def get_final_detection_info(self, final_contour, final_mask, detection):
        """Lấy thông tin chi tiết về vùng phát hiện cuối cùng"""
        if final_contour is None or len(final_contour) == 0:
            return None
        
        # Tính bounding box của final contour
        x_coords = final_contour[:, 0]
        y_coords = final_contour[:, 1]
        
        min_x, max_x = x_coords.min(), x_coords.max()
        min_y, max_y = y_coords.min(), y_coords.max()
        
        # Tính tâm
        M = cv2.moments(final_contour)
        if M["m00"] == 0:
            center_x = (min_x + max_x) // 2
            center_y = (min_y + max_y) // 2
        else:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
        
        # Tính diện tích và chu vi
        area = cv2.contourArea(final_contour)
        perimeter = cv2.arcLength(final_contour, True)
        
        # Tính độ tròn (circularity)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        # Tính ellipse fitting
        if len(final_contour) >= 5:
            ellipse = cv2.fitEllipse(final_contour)
            major_axis = max(ellipse[1])
            minor_axis = min(ellipse[1])
            aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 1
        else:
            aspect_ratio = 1
            ellipse = None
        
        # Tính solidity
        hull = cv2.convexHull(final_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        return {
            'final_contour': final_contour,
            'final_mask': final_mask,
            'bounding_box': {
                'x_min': int(min_x),
                'y_min': int(min_y),
                'x_max': int(max_x),
                'y_max': int(max_y),
                'width': int(max_x - min_x),
                'height': int(max_y - min_y)
            },
            'center': {
                'x': center_x,
                'y': center_y
            },
            'properties': {
                'area': float(area),
                'perimeter': float(perimeter),
                'circularity': float(circularity),
                'aspect_ratio': float(aspect_ratio),
                'solidity': float(solidity),
                'ellipse': ellipse
            },
            'original_detection': detection
        }
    def predict_quality(self, features: np.ndarray) -> Dict:
        """Improved prediction with better device handling"""
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Move model to device
            self.dgcnn_model = self.dgcnn_model.to(device)
            
            # Prepare features
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
            
            # Sampling/padding to fixed size (2048 points)
            max_points = 2048
            if features_tensor.size(1) > max_points:
                # Random sampling for large point clouds
                indices = torch.randperm(features_tensor.size(1))[:max_points]
                features_tensor = features_tensor[:, indices, :]
            elif features_tensor.size(1) < max_points:
                # Padding for small point clouds
                n_pad = max_points - features_tensor.size(1)
                pad_indices = torch.randperm(features_tensor.size(1))[:n_pad % features_tensor.size(1)]
                if n_pad > features_tensor.size(1):
                    # Multiple rounds of padding
                    repeat_times = n_pad // features_tensor.size(1)
                    remainder = n_pad % features_tensor.size(1)
                    pad_features = features_tensor.repeat(1, repeat_times, 1)
                    if remainder > 0:
                        extra_pad = features_tensor[:, :remainder, :]
                        pad_features = torch.cat([pad_features, extra_pad], dim=1)
                else:
                    pad_features = features_tensor[:, pad_indices, :]
                features_tensor = torch.cat([features_tensor, pad_features], dim=1)
            
            # Predict
            self.dgcnn_model.eval()
            with torch.no_grad():
                outputs = self.dgcnn_model(features_tensor)
                probabilities = F.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, predicted_class].item()
            
            # Create result dictionary
            class_probabilities = {}
            for i, class_name in self.class_names.items():
                class_probabilities[class_name] = probabilities[0, i].item()
            
            return {
                'predicted_class': predicted_class,
                'predicted_label': self.class_names[predicted_class],
                'confidence': confidence,
                'class_probabilities': class_probabilities
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None
    # Thêm hàm này vào bên trong class AppleQualityDetector
    def register_globally_and_refine(self, source_pc, target_pc, voxel_size=0.01):
        """
        Thực hiện đăng ký toàn cục bằng RANSAC và tinh chỉnh bằng ICP.
        """
        # Downsample để tăng tốc
        source_down = source_pc.voxel_down_sample(voxel_size)
        target_down = target_pc.voxel_down_sample(voxel_size)

        # Tính toán normals
        source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

        # --- Bước 1: Đăng ký toàn cục với FPFH + RANSAC ---
        # Tính toán đặc trưng FPFH
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))

        distance_threshold_ransac = voxel_size * 1.5
        ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            distance_threshold_ransac,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold_ransac)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
        
        print(f"    -> Global Registration (RANSAC) Fitness: {ransac_result.fitness:.3f}")

        # --- Bước 2: Tinh chỉnh cục bộ với ICP ---
        # Sử dụng kết quả của RANSAC làm gợi ý ban đầu cho ICP
        distance_threshold_icp = voxel_size * 0.8
        icp_result = o3d.pipelines.registration.registration_icp(
            source_down, target_down, distance_threshold_icp,
            ransac_result.transformation, # Dùng kết quả của RANSAC làm gợi ý
            o3d.pipelines.registration.TransformationEstimationPointToPlane())

        return icp_result
    
    def detect_apple(self, rgb_frame: np.ndarray, scale_factor: float = 1.1) -> Optional[Tuple[int, int, int, int]]:
        """
        Phát hiện quả táo sử dụng quy trình nâng cao từ yolo_detect_tao.txt.
        Quy trình này tạo nhiều prompt, đánh giá nhiều mask và chọn ra contour tốt nhất.
        """
        if self.sam_predictor is None or self.yolo_model is None:
            logger.error("Mô hình SAM hoặc YOLO không có sẵn. Không thể thực hiện phát hiện.")
            return None

        height, width, _ = rgb_frame.shape
        logger.info("  -> Bắt đầu quy trình phát hiện và phân đoạn nâng cao...")

        # 1. Phát hiện đối tượng ban đầu với YOLO
        try:
            # YOLO model của Roboflow yêu cầu đường dẫn file, nên ta cần lưu ảnh tạm thời
            temp_image_path = "temp_apple_detect.jpg"
            cv2.imwrite(temp_image_path, rgb_frame)
            
            # Gọi predict từ model YOLO [cite: 177]
            result = self.yolo_model.predict(temp_image_path, confidence=15, overlap=30).json()
            os.remove(temp_image_path) # Xóa file tạm

            if not result.get('predictions'):
                logger.info("  -> KẾT QUẢ YOLO: Không tìm thấy đối tượng nào.")
                return None

            # Lấy detection tốt nhất [cite: 178]
            detection = max(result["predictions"], key=lambda x: x["confidence"])
            logger.info(f"  -> YOLO phát hiện đối tượng với confidence: {detection['confidence']:.3f}")

        except Exception as e:
            logger.error(f"  -> Lỗi khi gọi YOLO hoặc xử lý kết quả: {e}")
            return None

        # 2. Tạo các prompts nâng cao 
        prompts = self.create_enhanced_prompts(detection, rgb_frame.shape)

        # 3. Phân đoạn với SAM sử dụng tất cả prompts 
        sam_results = self.segment_with_sam(rgb_frame, prompts)
        if not sam_results:
            logger.warning("  -> SAM không tạo ra được mask nào.")
            return None

        # 4. Tìm mask tốt nhất bằng thuật toán đánh giá cải tiến 
        best_mask, mask_info = self.find_best_mask_enhanced(sam_results, detection, rgb_frame.shape)
        if best_mask is None:
            logger.warning("  -> Không tìm thấy mask phù hợp sau khi đánh giá.")
            # Fallback: trả về box YOLO ban đầu nếu không có mask tốt
            x, y, w, h = detection['x'], detection['y'], detection['width'], detection['height']
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)
            return (x1, y1, x2, y2)

        # 5. Tạo contour cuối cùng với scaling thích ứng [cite: 260]
        _, final_contour, _ = self.create_adaptive_final_contour(best_mask, detection, adaptive_scale=True)
        if final_contour is None:
            logger.warning("  -> Không thể tạo contour cuối cùng.")
            return None
            
        # 6. Lấy bounding box từ contour cuối cùng
        final_x, final_y, final_w, final_h = cv2.boundingRect(final_contour)
        fx1, fy1, fx2, fy2 = final_x, final_y, final_x + final_w, final_y + final_h

        # Giới hạn tọa độ trong kích thước ảnh
        fx1 = max(0, fx1)
        fy1 = max(0, fy1)
        fx2 = min(width, fx2)
        fy2 = min(height, fy2)
        
        logger.info(f"  -> Bounding box cuối cùng từ mask tốt nhất: ({fx1}, {fy1}, {fx2}, {fy2})")
        return (fx1, fy1, fx2, fy2)

    def preprocess_pointcloud(self, pc: o3d.geometry.PointCloud, 
                                   max_points: int = 5000) -> o3d.geometry.PointCloud:
        """Memory efficient point cloud preprocessing"""
        # Early downsampling if too many points
        if len(pc.points) > max_points * 2:
            pc = pc.voxel_down_sample(voxel_size=0.01)
        
        # Remove outliers with adaptive parameters
        nb_neighbors = min(20, len(pc.points) // 10)
        if nb_neighbors >= 3:
            pc, _ = pc.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=2.0)
        
        # Final downsampling
        if len(pc.points) > max_points:
            pc = pc.voxel_down_sample(voxel_size=0.005)
        
        # Center and normalize
        if len(pc.points) > 0:
            pc_center = pc.get_center()
            pc.translate(-pc_center)
            
            # Scale to unit sphere
            points = np.asarray(pc.points)
            max_dist = np.linalg.norm(points, axis=1).max()
            if max_dist > 0:
                pc.scale(1.0 / max_dist, center=(0, 0, 0))
        
        return pc
    
    def extract_features(self, pc: o3d.geometry.PointCloud) -> np.ndarray:
        """Robust feature extraction with error handling"""
        try:
            # Estimate normals with adaptive parameters
            search_radius = 0.02
            max_nn = min(30, len(pc.points) // 5)
            
            if max_nn >= 3:
                pc.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=search_radius, max_nn=max_nn
                    )
                )
            
            points = np.asarray(pc.points)
            
            # Handle normals
            if pc.has_normals():
                normals = np.asarray(pc.normals)
                # Fix invalid normals
                invalid_normals = np.isnan(normals).any(axis=1) | (np.linalg.norm(normals, axis=1) < 1e-6)
                normals[invalid_normals] = [0, 0, 1]  # Default normal
            else:
                normals = np.tile([0, 0, 1], (len(points), 1))
            
            # Handle colors
            if pc.has_colors():
                colors = np.asarray(pc.colors)
            else:
                colors = np.ones_like(points) * 0.5
            
            # Compute curvature (simplified)
            curvature = np.linalg.norm(normals, axis=1)
            
            # Combine features: [x, y, z, nx, ny, nz, curvature, R, G, B]
            features = np.concatenate([
                points,                      # XYZ (3)
                normals,                     # Normals (3)
                curvature.reshape(-1, 1),    # Curvature (1)
                colors                       # RGB (3)
            ], axis=1)  # Total: 10 features
            
            # Validate features
            if np.isnan(features).any() or np.isinf(features).any():
                logger.warning("Invalid features detected, cleaning...")
                features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            # Return minimal features if extraction fails
            points = np.asarray(pc.points)
            features = np.concatenate([
                points,                                    # XYZ
                np.tile([0, 0, 1], (len(points), 1)),     # Default normals
                np.ones((len(points), 1)),                # Default curvature
                np.ones_like(points) * 0.5                # Default colors
            ], axis=1)
            return features


    def process_apple_multiview_improved(self, view_paths: List[Tuple[str, str]],
                                    visualize: bool = False,
                                    min_fitness: float = 0.3) -> Dict:
        """
        Process multi-view với registration cải tiến để tránh chồng lấp
        """
        try:
            # Bước 1: Load và làm sạch từng view
            all_pcs_raw = []
            original_indices = []
            view_centers = []  # Lưu tâm của mỗi view
            
            for i, (rgb_path, depth_path) in enumerate(view_paths):
                view_num = i + 1
                print(f"--- Đang tải View {view_num} ---")
                rgb = cv2.imread(rgb_path)
                if rgb is None:
                    logger.warning(f"Không thể đọc ảnh cho View {view_num}.")
                    continue
                depth = np.load(depth_path).astype(np.float32)

                bbox = self.detect_apple(rgb)
                if bbox is None:
                    logger.warning(f"!!! YOLO thất bại ở View {view_num}. Sử dụng box mặc định ở trung tâm.")
                    h, w, _ = rgb.shape
                    cx, cy = w // 2, h // 2
                    box_w, box_h = int(w * 0.6), int(h * 0.6)
                    x1 = max(0, cx - box_w // 2)
                    y1 = max(0, cy - box_h // 2)
                    x2 = min(w, cx + box_w // 2)
                    y2 = min(h, cy + box_h // 2)
                    bbox = (x1, y1, x2, y2)

                pc = self.create_pointcloud_from_rgbd(rgb, depth, bbox)
                if len(pc.points) > 100:
                    # Làm sạch outliers
                    pc_clean, _ = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
                    
                    # Lưu tâm gốc trước khi center
                    center = pc_clean.get_center()
                    view_centers.append(center)
                    
                    # Center point cloud về origin
                    pc_clean.translate(-center)
                    
                    all_pcs_raw.append(pc_clean)
                    original_indices.append(i)

            if len(all_pcs_raw) < 2:
                raise ValueError(f"Cần ít nhất 2 view hợp lệ để hợp nhất, chỉ có {len(all_pcs_raw)}.")

            # Bước 2: Đăng ký với initial transformation dựa trên góc chụp
            # Giả sử 4 view được chụp cách đều 90 độ quanh táo
            merged_pc = copy_pointcloud(all_pcs_raw[0])
            successful_merges = 1
            
            print(f"\n📸 Sử dụng View {original_indices[0] + 1} làm reference.")
            
            for i in range(1, len(all_pcs_raw)):
                source_pc = all_pcs_raw[i]
                source_view_index = original_indices[i]
                
                print(f"   Đăng ký View {source_view_index + 1} vào kết quả đã hợp nhất...")
                
                # Tạo initial transformation dựa trên góc view
                # Giả sử các view được chụp theo thứ tự: front (0°), right (90°), back (180°), left (270°)
                angle = source_view_index * np.pi / 2  # Convert to radians
                
                # Tạo rotation matrix quanh trục Y (vertical axis)
                initial_transform = np.eye(4)
                initial_transform[:3, :3] = np.array([
                    [np.cos(angle), 0, np.sin(angle)],
                    [0, 1, 0],
                    [-np.sin(angle), 0, np.cos(angle)]
                ])
                
                # Apply initial transformation
                source_pc_init = copy_pointcloud(source_pc)
                source_pc_init.transform(initial_transform)
                
                # Fine-tune với ICP
                print(f"    -> Applying initial rotation: {angle * 180 / np.pi}°")
                print("    -> Estimating normals for ICP...")
                source_pc_init.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
                merged_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
                # Sử dụng ICP với initial guess
                icp_result = o3d.pipelines.registration.registration_icp(
                    source_pc_init, merged_pc,
                    max_correspondence_distance=0.05,  # Điều chỉnh dựa trên scale của object
                    init=np.eye(4),
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                        relative_fitness=1e-6,
                        relative_rmse=1e-6,
                        max_iteration=50
                    )
                )
                
                # Combine transformations
                final_transformation = np.dot(icp_result.transformation, initial_transform)
                fitness = icp_result.fitness
                
                print(f"    -> ICP Fitness: {fitness:.3f}")
                
                # Validate registration bằng cách kiểm tra overlap
                if fitness > min_fitness:
                    # Transform và merge
                    source_transformed = copy_pointcloud(source_pc)
                    source_transformed.transform(final_transformation)
                    
                    # Kiểm tra overlap trước khi merge
                    dists = source_transformed.compute_point_cloud_distance(merged_pc)
                    dists = np.asarray(dists)
                    overlap_ratio = np.sum(dists < 0.02) / len(dists)
                    
                    print(f"    -> Overlap ratio: {overlap_ratio:.2%}")
                    
                    if overlap_ratio > 0.3:  # Ít nhất 30% overlap
                        # Merge với voxel downsampling để tránh duplicate points
                        temp_merged = merged_pc + source_transformed
                        merged_pc = temp_merged.voxel_down_sample(voxel_size=0.002)
                        
                        successful_merges += 1
                        print(f"    ✅ View {source_view_index + 1} merged successfully")
                    else:
                        print(f"    ⚠️ View {source_view_index + 1} - Low overlap, trying alternative method...")
                        
                        # Thử phương pháp RANSAC-based registration
                        result = self.register_globally_and_refine(source_pc, merged_pc)
                        if result.fitness > min_fitness:
                            source_transformed = copy_pointcloud(source_pc)
                            source_transformed.transform(result.transformation)
                            temp_merged = merged_pc + source_transformed
                            merged_pc = temp_merged.voxel_down_sample(voxel_size=0.002)
                            successful_merges += 1
                            print(f"    ✅ View {source_view_index + 1} merged with RANSAC")
                        else:
                            print(f"    ❌ View {source_view_index + 1} registration failed")
                else:
                    print(f"    ❌ View {source_view_index + 1} registration failed (low fitness)")

            print(f"\n🔄 Đã hợp nhất {successful_merges}/{len(all_pcs_raw)} views.")
            
            # Bước 3: Post-processing
            # Remove outliers cuối cùng
            if len(merged_pc.points) > 0:
                merged_pc, _ = merged_pc.remove_statistical_outlier(
                    nb_neighbors=50, std_ratio=1.0
                )
            
            # Chuẩn hóa cuối cùng
            final_pc = self.preprocess_pointcloud(merged_pc, max_points=10000)

            # Extract features và predict
            features = self.extract_features(final_pc)
            prediction = self.predict_quality(features) if self.dgcnn_model else None

            if visualize:
                self.visualize_multiview_result_enhanced(all_pcs_raw, final_pc, original_indices)

            return {
                'success': True,
                'point_cloud_size': len(final_pc.points),
                'features': features,
                'prediction': prediction,
                'merged_views': successful_merges
            }

        except Exception as e:
            logger.error(f"Lỗi trong process_apple_multiview: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}


    def visualize_multiview_result_enhanced(self, original_pcs, merged_pc, original_indices):
        # Visualize individual views trong subplots
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]  # Red, Green, Blue, Yellow
        
        # Cũng hiển thị với Open3D để có thể tương tác
        print("\n🎨 Hiển thị interative 3D:")
        print("  - Sử dụng chuột để xoay")
        print("  - Scroll để zoom")
        print("  - Các màu khác nhau = các view gốc")
        print("  - Nhấn 'q' để đóng cửa sổ")
        
        # Color code các view gốc
        colored_pcs = []
        for i, (pc, idx) in enumerate(zip(original_pcs, original_indices)):
            pc_colored = copy_pointcloud(pc)
            pc_colored.paint_uniform_color(colors[idx])
            
            # Apply transformation để hiển thị đúng vị trí
            angle = idx * np.pi / 2
            transform = np.eye(4)
            transform[:3, :3] = np.array([
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)]
            ])
            pc_colored.transform(transform)
            colored_pcs.append(pc_colored)
        
        # Thêm merged point cloud
        merged_colored = copy_pointcloud(merged_pc)
        merged_colored.translate([0, 0.3, 0])  # Dịch lên trên một chút
        
        # Tạo coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        
        # Show all
        o3d.visualization.draw_geometries(
            colored_pcs + [merged_colored, coord_frame],
            window_name="Multi-view Registration Result",
            width=1024,
            height=768
        )


    def debug_individual_registrations(self, all_pcs):
        """
        Debug từng cặp registration để hiểu vấn đề
        """
        print("\n🔍 Debug: Kiểm tra từng cặp registration")
        
        for i in range(len(all_pcs)):
            for j in range(i+1, len(all_pcs)):
                print(f"\n--- Thử registration View {i+1} với View {j+1} ---")
                
                # Clone point clouds
                pc1 = copy_pointcloud(all_pcs[i])
                pc2 = copy_pointcloud(all_pcs[j])
                
                # Paint different colors
                pc1.paint_uniform_color([1, 0, 0])  # Red
                pc2.paint_uniform_color([0, 1, 0])  # Green
                
                # Show before registration
                print("Trước registration:")
                o3d.visualization.draw_geometries(
                    [pc1, pc2],
                    window_name=f"Before: View {i+1} (Red) vs View {j+1} (Green)"
                )
                
                # Try registration
                result = self.register_globally_and_refine(pc1, pc2)
                print(f"Fitness: {result.fitness:.3f}")
                
                # Show after registration
                pc1_transformed = copy_pointcloud(pc1)
                pc1_transformed.transform(result.transformation)
                
                print("Sau registration:")
                o3d.visualization.draw_geometries(
                    [pc1_transformed, pc2],
                    window_name=f"After: View {i+1} (Red) vs View {j+1} (Green)"
                )
                

# =================== 5. TRAINING MODULE ===================
def train_model(dataset_dir: str, epochs: int = 50, batch_size: int = 8, 
                        k: int = 20, ball_radius: float = 0.1, use_hybrid: bool = True,
                        use_gat: bool = True, early_stopping_patience: int = 10):
    """Enhanced training with early stopping and better validation"""
    
    # Load and validate dataset
    features_list = []
    labels_list = []
    
    for file_path in Path(dataset_dir).glob("*.npz"):
        try:
            data = np.load(file_path)
            features = data['features']
            label = data['label'].item()
            
            # Validate data
            if features.shape[1] != 10:
                logger.warning(f"Invalid feature dimension in {file_path}: {features.shape}")
                continue
            if not (0 <= label <= 3):
                logger.warning(f"Invalid label in {file_path}: {label}")
                continue
                
            features_list.append(features)
            labels_list.append(label)
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            continue
    
    if len(features_list) < 10:
        print(f"❌ Insufficient training data: {len(features_list)} samples. Need at least 10.")
        return
    
    print(f"✅ Loaded {len(features_list)} valid samples")
    
    # Check class distribution
    unique_labels, counts = np.unique(labels_list, return_counts=True)
    print("📊 Class distribution:")
    class_names = ["Normal", "Bruised", "Cracked", "Rotten"]
    for label, count in zip(unique_labels, counts):
        print(f"   {class_names[label]}: {count} samples")
    
    # Split with stratification if possible
    try:
                # Chia thành 70% train, 15% val, 15% test
        X_temp, X_test, y_temp, y_test = train_test_split(
            features_list, labels_list, test_size=0.15, random_state=42, stratify=labels_list)

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp)  # 0.1765 ≈ 15/85

    # ĐOẠN CODE MỚI ĐÃ SỬA LỖI
    except ValueError:
        # Fallback without stratification if some classes have too few samples
        print("⚠️ Using non-stratified split due to class imbalance")
        
        # Chia thành 2 bước để đảm bảo có đủ 3 tập dữ liệu
        # Bước 1: Chia 85% cho (train + val), 15% cho test
        X_temp, X_test, y_temp, y_test = train_test_split(
            features_list, labels_list, test_size=0.15, random_state=42)

        # Bước 2: Chia 85% ở trên thành 70% train và 15% val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.1765, random_state=42)  # 0.1765 ≈ 15/85
    
    # Create datasets and loaders
    train_dataset = AppleDataset(X_train, y_train, augment=True)
    
    # TẮT augmentation cho tập validation và test
    val_dataset = AppleDataset(X_val, y_val, augment=False)
    test_dataset = AppleDataset(X_test, y_test, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Training on device: {device}")
    print(f"🔧 Config: hybrid={use_hybrid}, GAT={use_gat}, k={k}, radius={ball_radius}")
    
    model = DGCNN_Transformer(
        num_classes=4, input_dim=10, k=k, ball_radius=ball_radius,
        use_hybrid=use_hybrid, use_gat=use_gat
    ).to(device)
    
    # Optimizer and scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for better generalization
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    
    # Early stopping
    best_acc = 0
    patience_counter = 0
    train_losses = []
    train_accs = []
    test_accs = []
    
    print(f"🎯 Starting training for max {epochs} epochs with early stopping...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (batch_features, batch_labels) in enumerate(train_loader):
            batch_features = batch_features.to(device, non_blocking=True)
            batch_labels = batch_labels.squeeze().to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            try:
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_labels.size(0)
                train_correct += (predicted == batch_labels).sum().item()
                
            except RuntimeError as e:
                logger.error(f"Training error in batch {batch_idx}: {e}")
                continue
        
        # Validation phase
    
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device, non_blocking=True)
                batch_labels = batch_labels.squeeze().to(device, non_blocking=True)

                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()

        val_acc = 100 * val_correct / val_total if val_total > 0 else 0

        # Calculate metrics
        train_acc = 100 * train_correct / train_total if train_total > 0 else 0

        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else float('inf')
        
        # Store metrics
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)
        
        print(f'Epoch [{epoch+1:3d}/{epochs}] '
              f'Train Loss: {avg_train_loss:.4f} '
              f'Train Acc: {train_acc:6.2f}% '
              f'Val Acc: {val_acc:6.2f}% '
              f'LR: {optimizer.param_groups[0]["lr"]:.2e}')

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'config': {
                    'k': k,
                    'ball_radius': ball_radius,
                    'use_hybrid': use_hybrid,
                    'use_gat': use_gat
                }
            }, 'best_dgcnn_model.pt')
            print(f'✅ New best model saved! Val Acc: {best_acc:.2f}%')
            patience_counter = 0
        else:
            patience_counter += 1

        scheduler.step(val_acc)

        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"🛑 Early stopping at epoch {epoch+1}")
            break
    # =================== ĐÁNH GIÁ TRÊN TẬP TEST ===================
    print("\n----------------------------------------------------")
    print("🏁 Bắt đầu đánh giá trên tập dữ liệu Test...")

    # Tải lại mô hình tốt nhất đã lưu
    best_model_path = 'best_dgcnn_model.pt'
    if os.path.exists(best_model_path):
        # Khởi tạo lại một mô hình mới để chắc chắn không bị ảnh hưởng
        final_model = DGCNN_Transformer(
            num_classes=4, input_dim=10, k=k, ball_radius=ball_radius,
            use_hybrid=use_hybrid, use_gat=use_gat
        ).to(device)
        
        # Tải trọng số tốt nhất
        checkpoint = torch.load(best_model_path, map_location=device)
        final_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Đã tải mô hình tốt nhất từ epoch {checkpoint['epoch']+1} với Val Acc: {checkpoint['best_acc']:.2f}%")

        # Bắt đầu đánh giá
        final_model.eval()
        test_correct = 0
        test_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.squeeze().to(device)

                outputs = final_model(batch_features)
                _, predicted = torch.max(outputs.data, 1)

                test_total += batch_labels.size(0)
                test_correct += (predicted == batch_labels).sum().item()
                
                # Lưu lại dự đoán và nhãn thật để xem báo cáo chi tiết
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())

        # Tính toán và in ra độ chính xác cuối cùng
        final_test_acc = 100 * test_correct / test_total if test_total > 0 else 0
        print(f"\n🏆 === Độ chính xác cuối cùng trên tập Test: {final_test_acc:.2f}% === 🏆")

        # In ra báo cáo phân loại chi tiết
        class_names = ["Normal", "Bruised", "Cracked", "Rotten"]
        print("\n📊 Báo cáo phân loại chi tiết trên tập Test:")
        print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    else:
        print(f"❌ Không tìm thấy file mô hình '{best_model_path}'. Bỏ qua bước đánh giá trên tập test.")
    
    print("----------------------------------------------------")
    print(f'\n🎯 Training completed!')
    print(f'🏆 Best Test Accuracy: {best_acc:.2f}%')
    
    # Plot results
    #plot_training_results(train_losses, train_accs, test_accs)
    
    return model, best_acc
# =================== PLOT TRAINING ===================
'''def plot_training_results(train_losses, train_accs, test_accs):
    """Plot training results"""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, 'b-', linewidth=2)
    plt.title('Training Loss', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, 'g-', label='Train Accuracy', linewidth=2)
    plt.plot(test_accs, 'r-', label='Test Accuracy', linewidth=2)
    plt.title('Training Progress', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(np.array(test_accs) - np.array(train_accs), 'm-', linewidth=2)
    plt.title('Generalization Gap', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Test Acc - Train Acc (%)')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()'''

# =================== 6. LABELING HELPER ===================

def label_dataset_interactive(input_dir: str, output_dir: str,
                              rf_api_key: str, rf_project: str, rf_version: int,
                              sam_path: str):
    """Gán nhãn dataset một cách tương tác"""
    detector = AppleQualityDetector(
        rf_api_key=rf_api_key,
        rf_project=rf_project,
        rf_version=rf_version,
        sam_checkpoint_path=sam_path # Sử dụng biến được truyền vào
    )
    os.makedirs(output_dir, exist_ok=True)
    
    class_names = {
        '0': "Normal",
        '1': "Bruised", 
        '2': "Cracked",
        '3': "Rotten",
    }
    
    print("\n🏷️  Bắt đầu gán nhãn dataset")
    print("Nhãn có sẵn:")
    for key, value in class_names.items():
        print(f"  {key}: {value}")
    print("\nNhấn 's' để skip, 'q' để quit")
    
    apple_id = 1
    labeled_count = 0
    
    while True:
        view_paths = []
        for view_id in range(1, 5):  # 4 views
            rgb_path = os.path.join(input_dir, f"apple{apple_id:02d}_view{view_id:02d}_rgb.jpg")
            depth_path = os.path.join(input_dir, f"apple{apple_id:02d}_view{view_id:02d}_depth.npy")
            
            if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
                print(f"\n✅ Hoàn thành! Đã gán nhãn {labeled_count} quả táo.")
                return
            
            view_paths.append((rgb_path, depth_path))
        
        print(f"\n🍎 Xử lý quả táo {apple_id:02d}...")
        
        # Show preview images
        preview_images = []
        for rgb_path, _ in view_paths:
            rgb = cv2.imread(rgb_path)
            if rgb is not None:
                # Resize for preview
                h, w = rgb.shape[:2]
                preview_h = 200
                preview_w = int(w * preview_h / h)
                preview = cv2.resize(rgb, (preview_w, preview_h))
                preview_images.append(preview)
        
    
        if preview_images:
            combined = np.hstack(preview_images)
            plt.figure(figsize=(12, 4))
            plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
            plt.title(f'Apple {apple_id:02d} - All Views')
            plt.axis('off')
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(3)   # hoặc pause(0.001) nếu bạn muốn đóng bằng tay
            plt.close()
        # Process point cloud
        result = detector.process_apple_multiview_improved(view_paths, visualize=False)
        
        if result['success']:
            print(f"   Point cloud size: {result['point_cloud_size']} points")
            
            # Get label from user
            while True:
                label_input = input(f"   Nhập nhãn cho apple {apple_id:02d} (0-3, s=skip, q=quit): ").strip().lower()
                
                if label_input == 'q':
                    print(f"\n✅ Đã gán nhãn {labeled_count} quả táo.")
                    return
                elif label_input == 's':
                    print("   ⏭️  Skipped")
                    break
                elif label_input in class_names:
                    label = int(label_input)
                    features = result['features']
                    
                    # Save to dataset
                    save_path = os.path.join(output_dir, f"apple{apple_id:02d}.npz")
                    np.savez(save_path, features=features, label=label)
                    
                    print(f"   ✅ Đã lưu với nhãn: {class_names[label_input]}")
                    labeled_count += 1
                    break
                else:
                    print("   ❌ Nhãn không hợp lệ. Vui lòng nhập 0-3, s, hoặc q.")
        else:
            print(f"   ❌ Lỗi xử lý: {result['error']}")
            input("   Nhấn Enter để tiếp tục...")
        
        apple_id += 1
        

# =================== 7. MAIN PIPELINE ===================

def test_model_on_new_data(rf_api_key: str, rf_project: str, rf_version: int,
                           sam_path: str):
    """Test model với dữ liệu mới"""
    model_path = "best_dgcnn_model.pt"
    if not os.path.exists(model_path):
        print("❌ Không tìm thấy model đã train. Vui lòng train model trước.")
        return
    
    # Load detector with trained model
    detector = AppleQualityDetector(
        dgcnn_model_path=model_path,
        rf_api_key=rf_api_key,
        rf_project=rf_project,
        rf_version=rf_version,
        sam_checkpoint_path=sam_path # Sử dụng biến được truyền vào
    )
    
    input_dir = input("Thư mục chứa ảnh test: ").strip()
    if not os.path.exists(input_dir):
       print("❌ Thư mục không tồn tại. Vui lòng kiểm tra lại.")
       return

    apple_id = 1
    while True:
        # Tìm các view
        view_paths = []
        for view_id in range(1, 5):
            rgb_path = os.path.join(input_dir, f"apple{apple_id:02d}_view{view_id:02d}_rgb.jpg")
            depth_path = os.path.join(input_dir, f"apple{apple_id:02d}_view{view_id:02d}_depth.npy")
            if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
                break
            view_paths.append((rgb_path, depth_path))
        
        if len(view_paths) < 4:
            print(f"\n✅ Hoàn thành kiểm tra {apple_id - 1} quả táo.")
            break

        print(f"\n🍏 Đang kiểm tra apple {apple_id:02d}...")

        # Xử lý đa góc nhìn
        result = detector.process_apple_multiview_improved(view_paths, visualize=False)

        if result["success"]:
            prediction = result["prediction"]
            if prediction:
                print(f"   ✅ Dự đoán: {prediction['predicted_label']} "
                      f"(Confidence: {prediction['confidence']*100:.2f}%)")
                for cls, prob in prediction['class_probabilities'].items():
                    print(f"     - {cls}: {prob*100:.2f}%")
            else:
                print("   ⚠️ Không thể dự đoán.")
        else:
            print(f"   ❌ Lỗi xử lý: {result['error']}")
        
        input("   Nhấn Enter để tiếp tục tới quả táo tiếp theo...")
        apple_id += 1
if __name__ == "__main__":
    print("""
🎯 Apple Quality Detection System
Hệ thống phân loại chất lượng táo sử dụng DGCNN + Transformer + Multi-view Point Cloud

Vui lòng chọn chức năng muốn sử dụng:
1. Gán nhãn dữ liệu thủ công (interactive labeling)
2. Huấn luyện mô hình từ dữ liệu đã gán nhãn
3. Kiểm tra mô hình trên dữ liệu mới
0. Thoát
    """)
    RF_API_KEY = "0SwcE0IGoHTNCkcSfoqo"
    RF_PROJECT = "apple-detection-loeah"
    RF_VERSION = 1
    SAM_CHECKPOINT_PATH = "sam_vit_b_01ec64.pth"
    while True:
        choice = input("🔢 Nhập lựa chọn (0-3): ").strip()

        if choice == '1':
            input_dir = input("📂 Thư mục chứa ảnh raw multi-view: ").strip()
            output_dir = input("💾 Thư mục lưu dataset (.npz): ").strip()
            label_dataset_interactive(input_dir=input_dir, 
                output_dir=output_dir,
                rf_api_key=RF_API_KEY,
                rf_project=RF_PROJECT,
                rf_version=RF_VERSION,
                sam_path=SAM_CHECKPOINT_PATH
            )

        elif choice == '2':
            dataset_dir = input("📁 Thư mục chứa tập tin .npz: ").strip()
            try:
                epochs = int(input("🧠 Số epoch (default=50): ").strip() or "50")
                batch_size = int(input("📦 Batch size (default=8): ").strip() or "8")
                use_gat_input = input("🤖 Dùng GAT attention? (y/n, mặc định: y): ").strip().lower()
                use_gat = use_gat_input != 'n'
            except ValueError:
                print("⚠️ Giá trị không hợp lệ. Dùng mặc định.")
                epochs = 50
                batch_size = 8
                use_gat = True

            train_model(dataset_dir=dataset_dir, epochs=epochs, batch_size=batch_size, use_gat=use_gat)

        elif choice == '3':
            test_model_on_new_data(
                rf_api_key=RF_API_KEY,
                rf_project=RF_PROJECT,
                rf_version=RF_VERSION,
                sam_path=SAM_CHECKPOINT_PATH
            )

        elif choice == '0':
            print("👋 Tạm biệt!")
            break

        else:
            print("❌ Lựa chọn không hợp lệ. Vui lòng nhập số từ 0 đến 3.")

