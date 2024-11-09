import os
import numpy as np

def random_rotation(data, max_angle=30):
    """
    对数据进行随机旋转，旋转角度在 [-max_angle, max_angle] 之间。
    参数:
        data: 输入数据，形状为 (N, C, T, V, M)
        max_angle: 最大旋转角度
    返回:
        旋转后的数据
    """
    angle = np.random.uniform(-max_angle, max_angle) * np.pi / 180  # 随机角度转换为弧度
    cos_val, sin_val = np.cos(angle), np.sin(angle)

    # 假设绕 z 轴旋转
    rotation_matrix = np.array([[cos_val, -sin_val, 0],
                                [sin_val, cos_val, 0],
                                [0, 0, 1]])

    rotated_data = np.copy(data)
    for n in range(data.shape[0]):
        for t in range(data.shape[2]):
            for v in range(data.shape[3]):
                for m in range(data.shape[4]):
                    rotated_data[n, :, t, v, m] = np.dot(rotation_matrix, data[n, :, t, v, m])
    return rotated_data

def random_translation(data, max_shift=0.1):
    """
    对数据进行随机平移，平移幅度在 [-max_shift, max_shift] 之间。
    参数:
        data: 输入数据，形状为 (N, C, T, V, M)
        max_shift: 最大平移幅度
    返回:
        平移后的数据
    """
    translation = np.random.uniform(-max_shift, max_shift, size=(3,))  # 对 x, y, z 方向进行随机平移
    translated_data = data + translation[:, None, None, None]
    return translated_data

def random_scaling(data, min_scale=0.9, max_scale=1.1):
    """
    对数据进行随机缩放，缩放比例在 [min_scale, max_scale] 之间。
    参数:
        data: 输入数据，形状为 (N, C, T, V, M)
        min_scale: 最小缩放比例
        max_scale: 最大缩放比例
    返回:
        缩放后的数据
    """
    scale = np.random.uniform(min_scale, max_scale)
    scaled_data = data * scale
    return scaled_data

def random_jitter(data, noise_scale=0.02):
    """
    对数据添加随机抖动噪声。
    参数:
        data: 输入数据，形状为 (N, C, T, V, M)
        noise_scale: 噪声的尺度
    返回:
        添加噪声后的数据
    """
    noise = np.random.normal(0, noise_scale, size=data.shape)
    jittered_data = data + noise
    return jittered_data

def time_cropping(data, crop_ratio=0.8):
    """
    对数据进行时间剪切。
    参数:
        data: 输入数据，形状为 (N, C, T, V, M)
        crop_ratio: 剪切比例，决定保留的时间帧长度
    返回:
        剪切后的数据
    """
    T = data.shape[2]
    new_length = int(T * crop_ratio)
    start = np.random.randint(0, T - new_length)
    cropped_data = data[:, :, start:start + new_length, :, :]
    return cropped_data

def time_scaling(data, scale_factor=1.2):
    """
    对时间维度进行缩放。
    参数:
        data: 输入数据，形状为 (N, C, T, V, M)
        scale_factor: 缩放因子，大于 1 表示时间延长，小于 1 表示时间缩短
    返回:
        时间缩放后的数据
    """
    T = data.shape[2]
    new_length = int(T * scale_factor)
    scaled_data = np.zeros((data.shape[0], data.shape[1], new_length, data.shape[3], data.shape[4]))
    for n in range(data.shape[0]):
        for c in range(data.shape[1]):
            for v in range(data.shape[3]):
                for m in range(data.shape[4]):
                    scaled_data[n, c, :, v, m] = np.interp(
                        np.linspace(0, T - 1, new_length),
                        np.arange(T),
                        data[n, c, :, v, m]
                    )
    return scaled_data

# 示例：对 UAV-Human 数据集进行各种增广操作并保存增广后的数据集
if __name__ == "__main__":
    # 指定数据集路径
    input_file_path = "./data/original_data.npy"  # 替换为你的原始数据集路径
    output_directory = "./data/augmented/"  # 替换为保存增广数据的目录
    os.makedirs(output_directory, exist_ok=True)

    # 读取原始数据集
    data = np.load(input_file_path)
    
    # 应用各种增广方法
    data_rotated = random_rotation(data)
    data_translated = random_translation(data)
    data_scaled = random_scaling(data)
    data_jittered = random_jitter(data)
    data_cropped = time_cropping(data)
    data_time_scaled = time_scaling(data)
    
    # 保存增广后的数据集
    np.save(os.path.join(output_directory, "data_rotated.npy"), data_rotated)
    np.save(os.path.join(output_directory, "data_translated.npy"), data_translated)
    np.save(os.path.join(output_directory, "data_scaled.npy"), data_scaled)
    np.save(os.path.join(output_directory, "data_jittered.npy"), data_jittered)
    np.save(os.path.join(output_directory, "data_cropped.npy"), data_cropped)
    np.save(os.path.join(output_directory, "data_time_scaled.npy"), data_time_scaled)
    
    # 打印每种增广后的数据形状
    print("Original shape:", data.shape)
    print("Rotated shape:", data_rotated.shape)
    print("Translated shape:", data_translated.shape)
    print("Scaled shape:", data_scaled.shape)
    print("Jittered shape:", data_jittered.shape)
    print("Cropped shape:", data_cropped.shape)
    print("Time scaled shape:", data_time_scaled.shape)
