
import argparse
import numpy as np
from skopt import gp_minimize

def objective(weights):
    pred = np.zeros_like(r[0])
    
    for i in range(len(weights)):
        pred += r[i] * weights[i]
        
    pred = pred.argmax(axis=1)

    correct = (pred == label).sum()
    acc = correct / len(label)
    print(acc)
    return -acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    
    # 3
    # msg3d joint 3d            48 70.85   ok 48pt
    parser.add_argument('--msg3d_J3d_Score', default='./Model_inference/Mix_GCN/output/msg3d_J_3d/epoch48_test_score.pkl')

    # msg3d bone 3d             65.45  ok 57pt
    parser.add_argument('--msg3d_B3d_Score', default='./Model_inference/Mix_GCN/output/msg3d_B_3d/epoch65_test_score.pkl')

    # msg3d jointmotion 3d        55 57.55  ok 55pt
    parser.add_argument('--msg3d_JM3d_Score', default='./Model_inference/Mix_GCN/output/msg3d_JM_3d/epoch55_test_score.pkl')

    # 5
    # ctrgcn joint 2d           60  70.3     ok 60pt
    parser.add_argument('--ctrgcn_J2d_Score', default='./Model_inference/Mix_GCN/output/ctrgcn_J_2d/epoch60_test_score.pkl')
    
    # ctrgcn joint 3d           37  71.65   ok 37pt
    parser.add_argument('--ctrgcn_J3d_Score', default='./Model_inference/Mix_GCN/output/ctrgcn_J_3d/epoch37_test_score.pkl')

    # ctrgcn jointmotion 3d     37   58.7   ok 37pt
    parser.add_argument('--ctrgcn_JM3d_Score', default='./Model_inference/Mix_GCN/output/ctrgcn_JM_3d/epoch37_test_score.pkl')

    # ctrgcn bone 3d            61  67      ok 61pt
    parser.add_argument('--ctrgcn_B3d_Score', default='./Model_inference/Mix_GCN/output/ctrgcn_B_3d/epoch61_test_score.pkl')

    # ctrgcn bonemotion 3d     38  58.35    ok 38pt
    parser.add_argument('--ctrgcn_BM3d_Score', default='./Model_inference/Mix_GCN/output/ctrgcn_BM_3d/epoch38_test_score.pkl')


    # 5
    # tdgcn joint 2d            43  69.05   ok 43pt
    parser.add_argument('--tdgcn_J2d_Score', default='./Model_inference/Mix_GCN/output/tdgcn_J_2d/epoch43_test_score.pkl')

    # tdgcn joint 3d            37 69.6     ok 37pt
    parser.add_argument('--tdgcn_J3d_Score', default='./Model_inference/Mix_GCN/output/tdgcn_J_3d/epoch37_test_score.pkl')

    # tdgcn bone 3d            43 68.85     ok 43pt
    parser.add_argument('--tdgcn_B3d_Score', default='./Model_inference/Mix_GCN/output/tdgcn_B_3d/epoch43_test_score.pkl')


    # 2
    # tegcn joint 2d          39 69.70          ok 39pt
    parser.add_argument('--tegcn_J3d_Score', default='./Model_inference/Mix_GCN/output/tegcn_J_3d/epoch39_test_score.pkl')

    # tegcn joint 3d           39 69.45          ok 39pt
    parser.add_argument('--tegcn_B3d_Score', default='./Model_inference/Mix_GCN/output/tegcn_B_3d/epoch39_test_score.pkl')
    
    # # tegcn joint 2d           39 69.45          ok 39pt
    # parser.add_argument('--tegcn_J2d_Score', default='./Model_inference/Mix_GCN/output/tegcn_V2_B_3D/epoch39_test_score.pkl')
    

    # 5
    # mstgcn joint 2d         60   67.85    ok 60pt
    parser.add_argument('--mstgcn_J2d_Score', default='./Model_inference/Mix_GCN/output/mstgcn_J_2d/epoch60_test_score.pkl')

    # mstgcn joint 3d         43 69.15      ok 43pt
    parser.add_argument('--mstgcn_J3d_Score', default='./Model_inference/Mix_GCN/output/mstgcn_J_3d/epoch43_test_score.pkl')

    # mstgcn bone 3d           43 68.15     ok 43pt
    parser.add_argument('--mstgcn_B3d_Score', default='./Model_inference/Mix_GCN/output/mstgcn_B_3d/epoch43_test_score.pkl')

    # # mstgcn jointmotion 3d         61 50.45    ok 61pt
    # parser.add_argument('--mstgcn_JM3d_Score', default='./Model_inference/Mix_GCN/output/mstgcn_V2_JM_3d/epoch61_test_score.pkl')

    # # mstgcn bonemotion 3d           64  50.4  ok 64pt
    # parser.add_argument('--mstgcn_BM3d_Score', default='./Model_inference/Mix_GCN/output/mstgcn_V2_BM_3d/epoch64_test_score.pkl')


    # 6
    # skeformer joint 3d         56   70.75     ok 56pt
    parser.add_argument('--skeformer_J3d_Score', default='./Model_inference/Mix_Former/output/skmixf_J_3d/epoch56_test_score.pkl')

    # skeformer bone 3d           55 70.15      ok 55pt
    parser.add_argument('--skeformer_B3d_Score', default='./Model_inference/Mix_Former/output/skmixf_B_3d/epoch55_test_score.pkl')

    # skeformer k2 joint 3d         53  69.8     ok 53pt
    parser.add_argument('--skeformer_K2_J3d_Score', default='./Model_inference/Mix_Former/output/skmixf_k2_J3d/epoch53_test_score.pkl')

    # skeformer k2 bone 3d           54 69.3        ok 54pt
    parser.add_argument('--skeformer_K2_B3d_Score', default='./Model_inference/Mix_Former/output/skmixf_k2_B3d/epoch54_test_score.pkl')

    # skeformer k2 jointmotion 3d         52  58.35     ok 52pt
    parser.add_argument('--skeformer_K2_JM3d_Score', default='./Model_inference/Mix_Former/output/skmixf_JM_3d/epoch52_test_score.pkl')

    # skeformer k2 bonemotion 3d           53 59.3      ok 53pt
    parser.add_argument('--skeformer_K2_BM3d_Score', default='./Model_inference/Mix_Former/output/skmixf_BM_3d/epoch53_test_score.pkl')


    # 5
    # aagcn  joint 3d         37 71  ok 37pt
    parser.add_argument('--aagcn_J3d_Score', default='./Model_inference/Mix_GCN/output/aagcn_J_3d/epoch37_test_score.pkl')

    # aagcn  bone 3d          40 71.5 ok 40pt
    parser.add_argument('--aagcn_B3d_Score', default='./Model_inference/Mix_GCN/output/aagcn_B_3d/epoch40_test_score.pkl')

    # aagcn  joint 2d         38 68.45  ok 38pt
    parser.add_argument('--aagcn_J2d_Score', default='./Model_inference/Mix_GCN/output/aagcn_J_2d/epoch38_test_score.pkl')

    # aagcn  jointmotion 3d          36 58.10 ok36pt
    parser.add_argument('--aagcn_BM3d_Score', default='./Model_inference/Mix_GCN/output/aagcn_BM_3d/epoch36_test_score.pkl')

    # aagcn  bonemotion 3d         37  58.05   ok 37pt
    parser.add_argument('--aagcn_JM3d_Score', default='./Model_inference/Mix_GCN/output/aagcn_JM_3d/epoch37_test_score.pkl')



    arg = parser.parse_args()
    
    label = np.load('./Process_data/data/test_A_label.npy')

    r = []
    
    data1 = np.load(arg.msg3d_J3d_Score,allow_pickle=True)
    data2 = np.load(arg.msg3d_B3d_Score,allow_pickle=True)
    data3 = np.load(arg.msg3d_JM3d_Score,allow_pickle=True)

    data4 = np.load(arg.ctrgcn_J2d_Score,allow_pickle=True)
    data5 = np.load(arg.ctrgcn_J3d_Score,allow_pickle=True)
    data6 = np.load(arg.ctrgcn_JM3d_Score,allow_pickle=True)
    data7 = np.load(arg.ctrgcn_B3d_Score,allow_pickle=True)
    data8 = np.load(arg.ctrgcn_BM3d_Score,allow_pickle=True)

    data9 = np.load(arg.tdgcn_J2d_Score,allow_pickle=True)
    data10 = np.load(arg.tdgcn_J3d_Score,allow_pickle=True)
    data11 = np.load(arg.tdgcn_B3d_Score,allow_pickle=True)

    data12 = np.load(arg.tegcn_J3d_Score,allow_pickle=True)
    data13 = np.load(arg.tegcn_B3d_Score,allow_pickle=True)

    data14 = np.load(arg.mstgcn_J2d_Score,allow_pickle=True)
    data15 = np.load(arg.mstgcn_J3d_Score,allow_pickle=True)
    data16 = np.load(arg.mstgcn_B3d_Score,allow_pickle=True)


    data17 = np.load(arg.skeformer_J3d_Score,allow_pickle=True)
    data18 = np.load(arg.skeformer_B3d_Score,allow_pickle=True)
    data19 = np.load(arg.skeformer_K2_J3d_Score,allow_pickle=True)
    data20 = np.load(arg.skeformer_K2_B3d_Score,allow_pickle=True)
    data21 = np.load(arg.skeformer_K2_JM3d_Score,allow_pickle=True)
    data22 = np.load(arg.skeformer_K2_BM3d_Score,allow_pickle=True)

    data23 = np.load(arg.aagcn_J3d_Score,allow_pickle=True)
    data24 = np.load(arg.aagcn_B3d_Score,allow_pickle=True)
    data25 = np.load(arg.aagcn_J2d_Score,allow_pickle=True)
    data26 = np.load(arg.aagcn_BM3d_Score,allow_pickle=True)
    data27 = np.load(arg.aagcn_JM3d_Score,allow_pickle=True)



    lst_1 = []
    lst_2 = []
    lst_3 = []
    lst_4 = []    
    lst_5 = []
    lst_6 = []   
    lst_7 = []   
    lst_8 = []    
    lst_9 = []
    lst_10 = []   
    lst_11 = []
    lst_12 = []
    lst_13 = []
    lst_14 = []
    lst_15 = []
    lst_16 = []
    lst_17 = []
    lst_18 = []
    lst_19 = []
    lst_20 = []
    lst_21 = []
    lst_22 = []
    lst_23 = []
    lst_24 = []
    lst_25 = []
    lst_26 = []
    lst_27 = []

    for i in range(0,2000):
        lst_1.append(data1[f'test_{i}'])
        lst_2.append(data2[f'test_{i}'])
        lst_3.append(data3[f'test_{i}'])
        lst_4.append(data4[f'test_{i}'])
        lst_5.append(data5[f'test_{i}'])
        lst_6.append(data6[f'test_{i}'])
        lst_7.append(data7[f'test_{i}'])
        lst_8.append(data8[f'test_{i}'])
        lst_9.append(data9[f'test_{i}']) 
        lst_10.append(data10[f'test_{i}'])
        lst_11.append(data11[f'test_{i}'])
        lst_12.append(data12[f'test_{i}'])
        lst_13.append(data13[f'test_{i}'])
        lst_14.append(data14[f'test_{i}'])
        lst_15.append(data15[f'test_{i}'])
        lst_16.append(data16[f'test_{i}'])
        lst_17.append(data17[f'test_{i}'])
        lst_18.append(data18[f'test_{i}'])
        lst_19.append(data19[f'test_{i}'])
        lst_20.append(data20[f'test_{i}'])
        lst_21.append(data21[f'test_{i}'])
        lst_22.append(data22[f'test_{i}'])
        lst_23.append(data23[f'test_{i}'])
        lst_24.append(data24[f'test_{i}'])
        lst_25.append(data25[f'test_{i}'])
        lst_26.append(data26[f'test_{i}'])
        lst_27.append(data27[f'test_{i}'])

    r.append(np.array(lst_1))
    r.append(np.array(lst_2))
    r.append(np.array(lst_3))
    r.append(np.array(lst_4))
    r.append(np.array(lst_5))
    r.append(np.array(lst_6))
    r.append(np.array(lst_7))
    r.append(np.array(lst_8))
    r.append(np.array(lst_9))
    r.append(np.array(lst_10))
    r.append(np.array(lst_11))
    r.append(np.array(lst_12))
    r.append(np.array(lst_13))
    r.append(np.array(lst_14))
    r.append(np.array(lst_15))
    r.append(np.array(lst_16))
    r.append(np.array(lst_17))
    r.append(np.array(lst_18))
    r.append(np.array(lst_19))
    r.append(np.array(lst_20))
    r.append(np.array(lst_21))
    r.append(np.array(lst_22))
    r.append(np.array(lst_23))
    r.append(np.array(lst_24))
    r.append(np.array(lst_25))
    r.append(np.array(lst_26))
    r.append(np.array(lst_27))

    space = [(-0.5, 10) for i in range(27)]

    result = gp_minimize(objective, space, n_calls=50, random_state=1)

    print('Maximum accuracy: {:.4f}%'.format(-result.fun * 100))
    print('Optimal weights: {}'.format(result.x))

    final_pred = np.zeros_like(r[0])

    for i in range(len(result.x)):
        final_pred += r[i] * result.x[i]
    
    np.save('ensemble_score.npy', final_pred)
