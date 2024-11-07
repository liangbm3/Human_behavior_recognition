import numpy as np
import os
import argparse
import yaml

def save_pose_data(x_train, y_train, x_valid, y_valid, x_test, y_test, save_name):
    # 创建保存目录
    if not os.path.exists('./save_3d_pose'):
        os.makedirs('./save_3d_pose')
    
    # 保存训练和验证数据
    np.savez(f'./save_3d_pose/{save_name}.npz', 
             x_train=x_train, y_train=y_train, 
             x_test=x_valid, y_test=y_valid)

    # 保存测试数据
    np.savez(f'./save_3d_pose/{save_name}_test.npz', 
             x_test=x_test, y_test=y_test)

    print(f"Finished saving 3D pose data for {save_name}")

if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="Save 3D pose data.")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file.")

    args = parser.parse_args()
    
    # 读取 YAML 配置文件
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    # 加载数据
    x_train = np.load(config['x_train_path'], mmap_mode='r')
    y_train = np.load(config['y_train_path'])
    x_valid = np.load(config['x_valid_path'], mmap_mode='r')
    y_valid = np.load(config['y_valid_path'])
    x_test = np.load(config['x_test_path'], mmap_mode='r')
    y_test = np.load(config['y_test_path'])

    # 调用保存函数
    save_pose_data(x_train, y_train, x_valid, y_valid, x_test, y_test, config['save_name'])
