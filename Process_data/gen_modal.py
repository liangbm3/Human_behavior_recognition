import argparse
from tqdm import tqdm
import multiprocessing
from numpy.lib.format import open_memmap

parser = argparse.ArgumentParser(description='Dataset Preprocessing')
parser.add_argument('--use_mp', type=bool, default=False, help='use multi processing or not')
parser.add_argument('--modal', type=str, default='bone', help='modal type: bone, jmb, motion')

# UAV graph
graph = ((10, 8), (8, 6), (9, 7), (7, 5), (15, 13), (13, 11), (16, 14), (14, 12), 
         (11, 5), (12, 6), (11, 12), (5, 6), (5, 0), (6, 0), (1, 0), (2, 0), (3, 1), (4, 2))

sets = {'train', 'test', 'val'}  # updated set names
parts = {'joint', 'bone'}

# Bone generation function
def gen_bone(set_name):
    print(f'Processing {set_name} for bone data')
    data = open_memmap(f'./data/{set_name}_joint.npy', mode='r')  # Joint data for train/test/val
    N, C, T, V, M = data.shape
    fp_sp = open_memmap(f'./data/{set_name}_bone.npy', dtype='float32', mode='w+', shape=(N, 3, T, V, M))
    for v1, v2 in tqdm(graph):
        fp_sp[:, :, :, v1, :] = data[:, :, :, v1, :] - data[:, :, :, v2, :]

# Joint and bone merging function
def merge_joint_bone_data(set_name):
    print(f'Processing {set_name} for joint-bone merge')
    data_jpt = open_memmap(f'./data/{set_name}_joint.npy', mode='r')
    data_bone = open_memmap(f'./data/{set_name}_bone.npy', mode='r')
    N, C, T, V, M = data_jpt.shape
    data_jpt_bone = open_memmap(f'./data/{set_name}_joint_bone.npy', dtype='float32', mode='w+', shape=(N, 6, T, V, M))
    data_jpt_bone[:, :C, :, :, :] = data_jpt
    data_jpt_bone[:, C:, :, :, :] = data_bone

# Motion data generation function
def gen_motion(set_name, part):
    print(f'Processing {set_name} {part} for motion data')
    data = open_memmap(f'./data/{set_name}_{part}.npy', mode='r')  # Updated for joint/bone
    N, C, T, V, M = data.shape
    fp_sp = open_memmap(f'./data/{set_name}_{part}_motion.npy', dtype='float32', mode='w+', shape=(N, 3, T, V, M))
    for t in tqdm(range(T - 1)):
        fp_sp[:, :, t, :, :] = data[:, :, t + 1, :, :] - data[:, :, t, :, :]
    fp_sp[:, :, T - 1, :, :] = 0

if __name__ == '__main__':
    args = parser.parse_args()
    # Multiprocessing
    if args.use_mp:
        processes = []
        if args.modal == 'bone':   
            for set_name in sets:
                process = multiprocessing.Process(target=gen_bone, args=(set_name,))
                processes.append(process)
                process.start()
        elif args.modal == 'jmb':
            for set_name in sets:
                process = multiprocessing.Process(target=merge_joint_bone_data, args=(set_name,))
                processes.append(process)
                process.start()
        elif args.modal == 'motion':
            for set_name in sets:
                for part in parts:
                    process = multiprocessing.Process(target=gen_motion, args=(set_name, part))
                    processes.append(process)
                    process.start()
        else:
            raise ValueError('Invalid Modal')
        for process in processes:
            process.join()
    # Single-processing
    else:
        if args.modal == 'bone':   
            for set_name in sets:
                gen_bone(set_name)
        elif args.modal == 'jmb':
            for set_name in sets:
                merge_joint_bone_data(set_name)
        elif args.modal == 'motion':
            for set_name in sets:
                for part in parts:
                    gen_motion(set_name, part)
        else:
            raise ValueError('Invalid Modal')
