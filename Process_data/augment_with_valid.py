import numpy as np
import torch
from tqdm import tqdm   
import numpy as np

def save_processed_data(data, data_path='processed_data.npy'):

    # 将 PyTorch tensor 转换为 NumPy 数组
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()  # 如果在 GPU 上，需要先将数据移动到 CPU 上
    np.save(data_path, data)
    print(f"数据已保存到: {data_path}")



# augmented_list = [0, 1, 9, 10, 11, 12, 13, 14, 24, 25, 44, 45, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 90, 92, 93, 94, 95, 96, 97, 128, 131, 132, 133, 134, 135, 146, 147, 148, 152, 153, 154]

N, C, T, V, M = 16432, 3, 300, 17, 2 # N为取的样本数

train_data_path = "/data2/songxinshuai/ICMEW2024-Track10/Model_inference/Mix_GCN/data/train_joint.npy"
train_label_path = "/data2/songxinshuai/ICMEW2024-Track10/Model_inference/Mix_GCN/data/train_label.npy"

test_data_path = "/data2/songxinshuai/ICMEW2024-Track10/Model_inference/Mix_GCN/data/test_A_joint.npy"
test_label_path = "/data2/songxinshuai/ICMEW2024-Track10/Model_inference/Mix_GCN/data/test_A_label.npy"

def load_test_data():
    train_data_memmap = np.load(test_data_path, mmap_mode='r')
    train_label = np.load(test_label_path, mmap_mode='r')
    train_data = np.zeros((2000, C, 300, 17, 2), dtype=train_data_memmap.dtype)

    # 使用tqdm显示进度
    for i in tqdm(range(2000), desc='Loading test data'):
        train_data[i] = train_data_memmap[i]
    data = torch.tensor(train_data, dtype=torch.float32)  # 形状为 (C, T, V, M)
    
    mean = data.mean(dim=(1, 2, 3), keepdim=True)  # 形状为 (C, 1, 1, 1)
    std = data.std(dim=(1, 2, 3), keepdim=True)    # 形状为 (C, 1, 1, 1)
    std[std == 0] = 1e-8

    data_normalized = (data - mean) / std  # 形状为 (C, T, V, M)
    
    return data ,train_label[:N]


def load_train_data():
    train_data_memmap = np.load(train_data_path, mmap_mode='r')
    train_label = np.load(train_label_path, mmap_mode='r')
    train_data = np.zeros((N, C, 300, 17, 2), dtype=train_data_memmap.dtype)

    # 使用tqdm显示进度
    for i in tqdm(range(N), desc='Loading test data'):
        train_data[i] = train_data_memmap[i]
    data = torch.tensor(train_data, dtype=torch.float32)  # 形状为 (C, T, V, M)
    
    mean = data.mean(dim=(1, 2, 3), keepdim=True)  # 形状为 (C, 1, 1, 1)
    std = data.std(dim=(1, 2, 3), keepdim=True)    # 形状为 (C, 1, 1, 1)
    std[std == 0] = 1e-8

    data_normalized = (data - mean) / std  # 形状为 (C, T, V, M)
    
    return data_normalized ,train_label[:N]


graph = [
    (10, 8), (8, 6), (9, 7), (7, 5),
    (15, 13), (13, 11), (16, 14), (14, 12),
    (11, 5), (12, 6), (11, 12), (5, 6),
    (5, 0), (6, 0), (1, 0), (2, 0),
    (3, 1), (4, 2)
]


def smooth_data(data, window_size=3):
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    smoothed_data = data.clone()  # 使用 clone() 来创建数据的深拷贝
    N, C, T, V, M = data.shape
    
    # 在时间维度 T 上进行平滑
    for i in range(T - window_size):
        smoothed_data[:, :, i, :, :] = torch.mean(data[:, :, i:i+window_size, :, :], dim=2)  # 使用 torch.mean
    
    return smoothed_data


def scale_skeleton(skeleton_data, scale_range=(0.8, 1.2)):
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    scaled_skeleton = skeleton_data.clone()  # 保持原始数据不变
    # 假设前三个通道是 x, y, z 坐标
    scaled_skeleton[:, :3, :, :, :] *= scale_factor
    return scaled_skeleton


def skeleton_mirroring(data):
    """
    对骨骼点进行沿 z 轴的镜像翻转，交换左右对称的关节点并对 x 轴坐标进行翻转。
    
    参数:
    - data: 骨架数据，形状为 (N, C, T, V, M)，假设 C=6 且 C=0 表示 x 坐标，C=1 表示 y 坐标，C=2 表示 z 坐标。
    """
    N, C, T, V, M = data.shape
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    mirrored_data = data.clone()


    # 定义左右对称的关节点索引 (在 V 维度上)
    left_right_pairs = [
        (5, 6),   # 左肩和右肩
        (7, 8),   # 左肘和右肘
        (9, 10),  # 左手和右手
        (11, 12), # 左髋和右髋
        (13, 14), # 左膝和右膝
        (15, 16)  # 左脚和右脚
    ]

    # 在 V 维度上交换左右对称关节点的位置
    for left, right in left_right_pairs:
        mirrored_data[:, :, :, left, :], mirrored_data[:, :, :, right, :] = (
            data[:, :, :, right, :].clone(), data[:, :, :, left, :].clone()
        )

    # 在 C=0 维度上对 x 坐标进行镜像翻转
    mirrored_data[:, 0:2, :, :, :] = -mirrored_data[:, 0:2, :, :, :]

    return mirrored_data


def normalize_skeleton_data(data):
    N, C, T, V, M = data.shape
    normalized_data = data.clone()

    center_of_mass = torch.mean(normalized_data[:, :3, :, :, :], dim=3, keepdim=True)  # (N, C, T, 1, M)

    # Step 2: 平移骨骼点，使其中心点位于原点
    normalized_data[:, :3, :, :, :] -= center_of_mass

    return normalized_data




def apply_random_small_rotation(skeleton_data, max_angle=3):

    N, C, T, V, M = skeleton_data.shape

    # 确保 skeleton_data 是 torch.Tensor
    if isinstance(skeleton_data, np.ndarray):
        skeleton_data = torch.from_numpy(skeleton_data)

    # 生成随机的旋转角度，范围为 [-max_angle, max_angle]，并转换为弧度
    angle_x = np.radians(np.random.uniform(-max_angle*0.6, max_angle*0.6))
    angle_y = np.radians(np.random.uniform(-max_angle*0.6, max_angle*0.6))
    angle_z = np.radians(np.random.uniform(-max_angle, max_angle))

    # 旋转矩阵
    Rx = torch.tensor([[1, 0, 0],
                       [0, np.cos(angle_x), -np.sin(angle_x)],
                       [0, np.sin(angle_x), np.cos(angle_x)]], dtype=torch.float32)

    Ry = torch.tensor([[np.cos(angle_y), 0, np.sin(angle_y)],
                       [0, 1, 0],
                       [-np.sin(angle_y), 0, np.cos(angle_y)]], dtype=torch.float32)

    Rz = torch.tensor([[np.cos(angle_z), -np.sin(angle_z), 0],
                       [np.sin(angle_z), np.cos(angle_z), 0],
                       [0, 0, 1]], dtype=torch.float32)

    # 组合旋转矩阵：Rz * Ry * Rx
    rotation_matrix = Rz @ Ry @ Rx

    # 对每一帧的骨架数据进行旋转
    for i in range(N):
        for t in range(T):
            # 提取骨架的 x, y, z 坐标 (假设前3个通道是 x, y, z 坐标)
            coords = skeleton_data[i, :3, t, :, :].reshape(3, -1)  # (3, V * M)

            # 获取当前关节点的中心位置
            center_of_mass = torch.mean(coords, dim=1, keepdim=True)

            # 将关节点的坐标平移到原点进行旋转
            centered_coords = coords - center_of_mass

            # 进行旋转
            rotated_coords = rotation_matrix @ centered_coords

            # 旋转后将关节点位置平移回原位置
            rotated_coords = rotated_coords + center_of_mass

            # 将旋转后的坐标放回骨架数据中
            skeleton_data[i, :3, t, :, :] = rotated_coords.view(3, V, M)

    return skeleton_data



def mean_normalize_skeleton_data(data, standardize=False):
    
    N, C, T, V, M = data.shape

    # 克隆数据以避免原始数据被修改
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    normalized_data = data.clone()

    # Step 1: 计算每个样本的均值 (在 V 维度上求平均)，对 x, y, z 三个通道分别计算
    mean = torch.mean(normalized_data[:, :3, :, :, :], dim=3, keepdim=True)  # (N, C=3, T, 1, M)

    # Step 2: 对每个样本进行均值归一化
    normalized_data[:, :3, :, :, :] -= mean

    if standardize:
        # Step 3: 可选的标准差归一化，计算标准差并进行归一化
        std = torch.std(normalized_data[:, :3, :, :, :], dim=3, keepdim=True)
        std[std == 0] = 1  # 防止除以0的情况
        normalized_data[:, :3, :, :, :] /= std

    return normalized_data



left_right = [104,105,112,113,119,120]

difficult = [1,9,10,21,24,25,28,32,33,36,40,41,44,45,46,47,48,53,60,61,62,64,65,67,70,99,100,101,125,136,137]

double_action = [80,81,82,86,87,88,92,93,94,95,96,97,131,134,135,146,148]

inverse_action_pair = [3,4,6,7,11,12,13,14,15,16,22,23,34,35,38,39,49,50,51,52,80,81,82,84,85,102,103,106,107,108,109,115,116,144,145,153,154]

def generate_augmented_data_with_edges(train_data, train_labels, test_data,test_label):
    augmented_data = []
    augmented_labels = []

    for i in range(len(train_data)):
        # 对原始数据平滑，归一化操作
        sample = train_data[i:i+1]  # (1, C, T, V, M)
        label = train_labels[i]
        new_sample_1 = sample.clone()  # 使用 clone() 代替 copy()
        new_sample_1 = apply_random_small_rotation(new_sample_1)
        new_sample_1 = scale_skeleton(new_sample_1)
        new_sample_1 = smooth_data(new_sample_1)
        if isinstance(new_sample_1, np.ndarray):
            new_sample_1 = torch.from_numpy(new_sample_1)
        augmented_data.append(new_sample_1)
        augmented_labels.append(label)  
            
        if label in inverse_action_pair:
            new_sample_1 = sample.clone()  # 使用 clone() 代替 copy()
            new_sample_1 = apply_random_small_rotation(new_sample_1)
            new_sample_1 = scale_skeleton(new_sample_1)
            new_sample_1 = smooth_data(new_sample_1)
            if isinstance(new_sample_1, np.ndarray):
                    new_sample_1 = torch.from_numpy(new_sample_1)
            augmented_data.append(new_sample_1)
            augmented_labels.append(label)  
            
    for i in range(len(test_data)):
        # 对原始数据平滑，归一化操作
        sample = test_data[i:i+1]  # (1, C, T, V, M)
        label = test_label[i]
        new_sample_1 = sample.clone()  # 使用 clone() 代替 copy()
        new_sample_1 = apply_random_small_rotation(new_sample_1)
        new_sample_1 = scale_skeleton(new_sample_1)
        new_sample_1 = smooth_data(new_sample_1)

        if isinstance(new_sample_1, np.ndarray):
                new_sample_1 = torch.from_numpy(new_sample_1)
        augmented_data.append(new_sample_1)
        augmented_labels.append(label)  
        
        if label in difficult:
            for k in range(1):
                sample = test_data[i:i+1]  # (1, C, T, V, M)
                label = test_label[i]
                new_sample_1 = sample.clone()  # 使用 clone() 代替 copy()
                new_sample_1 = apply_random_small_rotation(new_sample_1)
                new_sample_1 = scale_skeleton(new_sample_1)
                new_sample_1 = smooth_data(new_sample_1)

                if isinstance(new_sample_1, np.ndarray):
                        new_sample_1 = torch.from_numpy(new_sample_1)
                augmented_data.append(new_sample_1)
                augmented_labels.append(label)  
      
    augmented_data = torch.cat(augmented_data, dim=0)
    augmented_labels = torch.tensor(augmented_labels)

    return augmented_data, augmented_labels


train_data_xyz,train_label_xyz = load_train_data()
# train_data_xy = train_data_xyz[:, :2, :, :, :]  # 选择 C=3 中的前两个通道 X 和 Y
test_data_xyz,test_label_xyz = load_test_data()


print(train_data_xyz.shape)
print(train_label_xyz.shape)

augmented_data , augmented_labels = generate_augmented_data_with_edges(train_data_xyz,train_label_xyz,
                                                                       test_data_xyz,test_label_xyz)

print(augmented_data.shape)

if isinstance(augmented_data, np.ndarray):
    augmented_data = torch.from_numpy(augmented_data)

if isinstance(augmented_labels,np.ndarray):
    augmented_labels = torch.from_numpy(augmented_labels)

# save_processed_data(train_data_xyz,'/data2/songxinshuai/behaviour/ICMEW2024-Track10/Model_inference/Mix_GCN/processed_data/train_joint_xyz.npy')
# save_processed_data(train_data_xy,'/data2/songxinshuai/behaviour/ICMEW2024-Track10/Model_inference/Mix_GCN/processed_data/train_joint_xy.npy')
save_processed_data(augmented_data,'/data2/songxinshuai/behaviour/ICMEW2024-Track10/Model_inference/Mix_GCN/processed_data/train_with_valid_data.npy')
save_processed_data(augmented_labels,'/data2/songxinshuai/behaviour/ICMEW2024-Track10/Model_inference/Mix_GCN/processed_data/train_with_valid_label.npy')


