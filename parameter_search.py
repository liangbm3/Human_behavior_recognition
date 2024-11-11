
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

    
    # 2
    # msg3d joint 3d            42 43.30%
    parser.add_argument('--msg3d_J3d_Score', default='./Model_inference/Mix_GCN/output/msg3d_J_3d/epoch42_test_score.pkl')

    # msg3d bone 3d            56 40.25%
    parser.add_argument('--msg3d_B3d_Score', default='./Model_inference/Mix_GCN/output/msg3d_B_3d/epoch56_test_score.pkl')

    # msg3d jointmotion 3d       51 33.2     
    parser.add_argument('--msg3d_JM3d_Score', default='./Model_inference/Mix_GCN/output/msg3d_JM_3d/epoch51_test_score.pkl')

    # msg3d bonemotion 3d      51 33.10%      
    parser.add_argument('--msg3d_BM3d_Score', default='./Model_inference/Mix_GCN/output/msg3d_BM_3d/epoch51_test_score.pkl')


    # 8
    # ctrgcn joint 2d           39 44.30%
    parser.add_argument('--ctrgcn_J2d_Score', default='./Model_inference/Mix_GCN/output/ctrgcn_J_2d/epoch39_test_score.pkl')

    # ctrgcn bone 2d           46 44.40%
    parser.add_argument('--ctrgcn_B2d_Score', default='./Model_inference/Mix_GCN/output/ctrgcn_B_2d/epoch46_test_score.pkl')

    # ctrgcn jointmotion 2d           36 35.05%
    parser.add_argument('--ctrgcn_JM2d_Score', default='./Model_inference/Mix_GCN/output/ctrgcn_JM_2d/epoch36_test_score.pkl')

    # ctrgcn bonemotion 2d           38 36.90%
    parser.add_argument('--ctrgcn_BM2d_Score', default='./Model_inference/Mix_GCN/output/ctrgcn_BM_2d/epoch38_test_score.pkl')
    
    # ctrgcn joint 3d         39 44.1  
    parser.add_argument('--ctrgcn_J3d_Score', default='./Model_inference/Mix_GCN/output/ctrgcn_J_3d/epoch39_test_score.pkl')

    # ctrgcn bone 3d           36 41.85%
    parser.add_argument('--ctrgcn_B3d_Score', default='./Model_inference/Mix_GCN/output/ctrgcn_B_3d/epoch36_test_score.pkl')

    # ctrgcn jointmotion 3d     38 35.65%
    parser.add_argument('--ctrgcn_JM3d_Score', default='./Model_inference/Mix_GCN/output/ctrgcn_JM_3d/epoch38_test_score.pkl')

    # ctrgcn bonemotion 3d    38 36.80%
    parser.add_argument('--ctrgcn_BM3d_Score', default='./Model_inference/Mix_GCN/output/ctrgcn_BM_3d/epoch38_test_score.pkl')


    # 2
    # tdgcn joint 3d            
    parser.add_argument('--tdgcn_J3d_Score', default='')

    # tdgcn bone 3d         37  40.8
    parser.add_argument('--tdgcn_B3d_Score', default='./Model_inference/Mix_GCN/output/tdgcn_B_3d/epoch37_test_score.pkl')

    

    # 2
    # tegcn joint 3d         36 43.2 
    parser.add_argument('--tegcn_J3d_Score', default='./Model_inference/Mix_GCN/output/tegcn_J_3d/epoch36_test_score.pkl')

    # tegcn bone 3d           38 43.8
    parser.add_argument('--tegcn_B3d_Score', default='./Model_inference/Mix_GCN/output/tegcn_B_3d/epoch38_test_score.pkl')
    
    

    # 6
    # mstgcn joint 2d         41 42.65%
    parser.add_argument('--mstgcn_J2d_Score', default='./Model_inference/Mix_GCN/output/mstgcn_J/epoch41_test_score.pkl')

    # mstgcn bone 2d         38 41.15%
    parser.add_argument('--mstgcn_B2d_Score', default='./Model_inference/Mix_GCN/output/mstgcn_B2d/epoch38_test_score.pkl')

    # mstgcn joint 3d         41 41.55
    parser.add_argument('--mstgcn_J3d_Score', default='./Model_inference/Mix_GCN/output/mstgcn_J_3d/epoch41_test_score.pkl')

    # mstgcn bone 3d          38 41.05
    parser.add_argument('--mstgcn_B3d_Score', default='./Model_inference/Mix_GCN/output/mstgcn_B_3d/epoch38_test_score.pkl')

    # mstgcn jointmotion 3d      
    parser.add_argument('--mstgcn_JM3d_Score', default='')

    # mstgcn bonemotion 3d     
    parser.add_argument('--mstgcn_BM3d_Score', default='')


    # 16
    # skeformer joint 2d        53 43.6
    parser.add_argument('--skeformer_J2d_Score', default='./Model_inference/Mix_Former/output/skmixf_J_2d/epoch53_test_score.pkl')

    # skeformer bone 2d         53 43.55
    parser.add_argument('--skeformer_B2d_Score', default='./Model_inference/Mix_Former/output/skmixf_B_2d/epoch53_test_score.pkl')

    # skeformer jointmotion 2d    retrain 1 33.75%
    parser.add_argument('--skeformer_JM2d_Score', default='./Model_inference/Mix_Former/output/skmixf_JM_2d/epoch1_test_score.pkl')

    # skeformer bonemotion 2d         retrain  1 34.40% 
    parser.add_argument('--skeformer_BM2d_Score', default='./Model_inference/Mix_Former/output/skmixf_BM_2d/epoch1_test_score.pkl')


    # skeformer joint 3d        52 44.15
    parser.add_argument('--skeformer_J3d_Score', default='./Model_inference/Mix_Former/output/skmixf_J_3d/epoch52_test_score.pkl')

    # skeformer bone 3d         55 44.35
    parser.add_argument('--skeformer_B3d_Score', default='./Model_inference/Mix_Former/output/skmixf_B_3d/epoch55_test_score.pkl')

    # skeformer jointmotion 3d        52 33.65
    parser.add_argument('--skeformer_JM3d_Score', default='./Model_inference/Mix_Former/output/skmixf_JM_3d/epoch52_test_score.pkl')

    # skeformer bonemotion 3d         51 33.7
    parser.add_argument('--skeformer_BM3d_Score', default='./Model_inference/Mix_Former/output/skmixf_BM_3d/epoch51_test_score.pkl')



    # skeformer k2 joint 3d       56 43.75%
    parser.add_argument('--skeformer_K2_J2d_Score', default='./Model_inference/Mix_Former/output/skmixf_k2_J2d/epoch56_test_score.pkl')

    # skeformer k2 bone 3d          57 43.25%
    parser.add_argument('--skeformer_K2_B2d_Score', default='./Model_inference/Mix_Former/output/skmixf_k2_B2d/epoch57_test_score.pkl')

    # skeformer k2 jointmotion 2d       53 35.90% 
    parser.add_argument('--skeformer_K2_JM2d_Score', default='./Model_inference/Mix_Former/output/skmixf_k2_JM2d/epoch53_test_score.pkl')

    # skeformer k2 bonemotion 2d      retrain   1 32.70%
    parser.add_argument('--skeformer_K2_BM2d_Score', default='./Model_inference/Mix_Former/output/skmixf_k2_BM2d/epoch1_test_score.pkl')


    # skeformer k2 joint 3d       52 44
    parser.add_argument('--skeformer_K2_J3d_Score', default='./Model_inference/Mix_Former/output/skmixf_k2_J3d/epoch52_test_score.pkl')

    # skeformer k2 bone 3d          58 43
    parser.add_argument('--skeformer_K2_B3d_Score', default='./Model_inference/Mix_Former/output/skmixf_k2_B3d/epoch58_test_score.pkl')

    # skeformer k2 jointmotion 3d      53 35.1
    parser.add_argument('--skeformer_K2_JM3d_Score', default='./Model_inference/Mix_Former/output/skmixf_k2_JM3d/epoch53_test_score.pkl')

    # skeformer k2 bonemotion 3d       55 34.95   
    parser.add_argument('--skeformer_K2_BM3d_Score', default='./Model_inference/Mix_Former/output/skmixf_k2_BM3d/epoch55_test_score.pkl')


    # 6
    # aagcn  joint 2d         42 42.90%
    parser.add_argument('--aagcn_J2d_Score', default='./Model_inference/Mix_GCN/output/aagcn_J_2d/epoch42_test_score.pkl')

    # aagcn  bone 2d         38 42.10%
    parser.add_argument('--aagcn_B2d_Score', default='./Model_inference/Mix_GCN/output/aagcn_B_2d/epoch38_test_score.pkl')

    # aagcn  joint 3d         38 43.80%
    parser.add_argument('--aagcn_J3d_Score', default='./Model_inference/Mix_GCN/output/aagcn_J_3d/epoch38_test_score.pkl')

    # aagcn  bone 3d         38 43.15%
    parser.add_argument('--aagcn_B3d_Score', default='./Model_inference/Mix_GCN/output/aagcn_B_3d/epoch38_test_score.pkl')

    # aagcn  jointmotion 3d      37 35.50%
    parser.add_argument('--aagcn_JM3d_Score', default='./Model_inference/Mix_GCN/output/aagcn_JM_3d/epoch37_test_score.pkl')

    # aagcn  bonemotion 3d         37 35.55%
    parser.add_argument('--aagcn_BM3d_Score', default='./Model_inference/Mix_GCN/output/aagcn_BM_3d/epoch37_test_score.pkl')

    


    arg = parser.parse_args()
    
    label = np.load('./Process_data/data/val_label.npy')

    r = []
    

    data1 = np.load(arg.ctrgcn_J2d_Score,allow_pickle=True)
    data2 = np.load(arg.ctrgcn_B2d_Score,allow_pickle=True)
    data3 = np.load(arg.ctrgcn_JM2d_Score,allow_pickle=True)
    data4 = np.load(arg.ctrgcn_BM2d_Score,allow_pickle=True)
    data5 = np.load(arg.ctrgcn_J3d_Score,allow_pickle=True)
    data6 = np.load(arg.ctrgcn_B3d_Score,allow_pickle=True)
    data7 = np.load(arg.ctrgcn_JM3d_Score,allow_pickle=True)
    data8 = np.load(arg.ctrgcn_BM3d_Score,allow_pickle=True)


    data9 = np.load(arg.mstgcn_J2d_Score,allow_pickle=True)
    data10 = np.load(arg.mstgcn_B2d_Score,allow_pickle=True)
    # data11 = np.load(arg.mstgcn_J3d_Score,allow_pickle=True)
    # data12 = np.load(arg.mstgcn_B3d_Score,allow_pickle=True)
    # data13 = np.load(arg.mstgcn_JM3d_Score,allow_pickle=True)
    # data14 = np.load(arg.mstgcn_BM3d_Score,allow_pickle=True)


    data15 = np.load(arg.skeformer_J2d_Score,allow_pickle=True)
    data16 = np.load(arg.skeformer_B2d_Score,allow_pickle=True)
    # data17 = np.load(arg.skeformer_JM2d_Score,allow_pickle=True)
    # data18 = np.load(arg.skeformer_BM2d_Score,allow_pickle=True)
    data19 = np.load(arg.skeformer_J3d_Score,allow_pickle=True)
    data20 = np.load(arg.skeformer_B3d_Score,allow_pickle=True)
    # data21 = np.load(arg.skeformer_JM3d_Score,allow_pickle=True)
    # data22 = np.load(arg.skeformer_BM3d_Score,allow_pickle=True)
    data23 = np.load(arg.skeformer_K2_J2d_Score,allow_pickle=True)
    data24 = np.load(arg.skeformer_K2_B2d_Score,allow_pickle=True)
    # data25 = np.load(arg.skeformer_K2_JM2d_Score,allow_pickle=True)
    # data26 = np.load(arg.skeformer_K2_BM2d_Score,allow_pickle=True)
    data27 = np.load(arg.skeformer_K2_J3d_Score,allow_pickle=True)
    data28 = np.load(arg.skeformer_K2_B3d_Score,allow_pickle=True)
    # data29 = np.load(arg.skeformer_K2_JM3d_Score,allow_pickle=True)
    # data30 = np.load(arg.skeformer_K2_BM3d_Score,allow_pickle=True)


    data31 = np.load(arg.msg3d_J3d_Score,allow_pickle=True)
    data32 = np.load(arg.msg3d_B3d_Score,allow_pickle=True)
    # data33 = np.load(arg.msg3d_JM3d_Score,allow_pickle=True)
    # data34 = np.load(arg.msg3d_BM3d_Score,allow_pickle=True)

    # tegcn j3d 已经训了
    # data35 = np.load(arg.tdgcn_J3d_Score,allow_pickle=True)
    # data36 = np.load(arg.tdgcn_B3d_Score,allow_pickle=True)


    # tehcn j3d 已经训了
    data37 = np.load(arg.tegcn_J3d_Score,allow_pickle=True)
    data38 = np.load(arg.tegcn_B3d_Score,allow_pickle=True)



    data39 = np.load(arg.aagcn_J2d_Score,allow_pickle=True)
    data40 = np.load(arg.aagcn_B2d_Score,allow_pickle=True)
    data41 = np.load(arg.aagcn_J3d_Score,allow_pickle=True)
    data42 = np.load(arg.aagcn_B3d_Score,allow_pickle=True)
    # data43 = np.load(arg.aagcn_JM3d_Score,allow_pickle=True)
    # data44 = np.load(arg.aagcn_BM3d_Score,allow_pickle=True)



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
    lst_28 = []
    lst_29 = []
    lst_30 = []
    lst_31 = []
    lst_32 = []
    lst_33 = []
    lst_34 = []
    lst_35 = []
    lst_36 = []
    lst_37 = []
    lst_38 = []
    lst_39 = []
    lst_40 = []
    lst_41 = []
    lst_42 = []
    lst_43 = []
    lst_44 = []


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
        # lst_11.append(data11[f'test_{i}'])
        # lst_12.append(data12[f'test_{i}'])
        # lst_13.append(data13[f'test_{i}'])
        # lst_14.append(data14[f'test_{i}'])
        lst_15.append(data15[f'test_{i}'])
        lst_16.append(data16[f'test_{i}'])
        # lst_17.append(data17[f'test_{i}'])
        # lst_18.append(data18[f'test_{i}'])
        lst_19.append(data19[f'test_{i}'])
        lst_20.append(data20[f'test_{i}'])
        # lst_21.append(data21[f'test_{i}'])
        # lst_22.append(data22[f'test_{i}'])
        lst_23.append(data23[f'test_{i}'])
        lst_24.append(data24[f'test_{i}'])
        # lst_25.append(data25[f'test_{i}'])
        # lst_26.append(data26[f'test_{i}'])
        lst_27.append(data27[f'test_{i}'])
        lst_28.append(data28[f'test_{i}'])
        # lst_29.append(data29[f'test_{i}'])
        # lst_30.append(data30[f'test_{i}'])
        # lst_31.append(data31[f'test_{i}'])
        # lst_32.append(data32[f'test_{i}'])
        # lst_33.append(data33[f'test_{i}'])
        # lst_34.append(data34[f'test_{i}'])
        # lst_35.append(data35[f'test_{i}'])
        # lst_36.append(data36[f'test_{i}'])
        lst_37.append(data37[f'test_{i}'])
        lst_38.append(data38[f'test_{i}'])
        lst_39.append(data39[f'test_{i}'])
        lst_40.append(data40[f'test_{i}'])
        lst_41.append(data41[f'test_{i}'])
        lst_42.append(data42[f'test_{i}'])
        # lst_43.append(data43[f'test_{i}'])
        # lst_44.append(data44[f'test_{i}'])


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
    # r.append(np.array(lst_11))
    # r.append(np.array(lst_12))
    # r.append(np.array(lst_13))
    # r.append(np.array(lst_14))
    r.append(np.array(lst_15))
    r.append(np.array(lst_16))
    # r.append(np.array(lst_17))
    # r.append(np.array(lst_18))
    r.append(np.array(lst_19))
    r.append(np.array(lst_20))
    # r.append(np.array(lst_21))
    # r.append(np.array(lst_22))
    r.append(np.array(lst_23))
    r.append(np.array(lst_24))
    # r.append(np.array(lst_25))
    # r.append(np.array(lst_26))
    r.append(np.array(lst_27))
    r.append(np.array(lst_28))
    # r.append(np.array(lst_29))
    # r.append(np.array(lst_30))
    # r.append(np.array(lst_31))
    # r.append(np.array(lst_32))
    # r.append(np.array(lst_33))
    # r.append(np.array(lst_34))
    # r.append(np.array(lst_35))
    # r.append(np.array(lst_36))
    r.append(np.array(lst_37))
    r.append(np.array(lst_38))
    r.append(np.array(lst_39))
    r.append(np.array(lst_40))
    r.append(np.array(lst_41))
    r.append(np.array(lst_42))
    # r.append(np.array(lst_43))
    # r.append(np.array(lst_44))

    space = [(0,1.2) for i in range(24)]

    result = gp_minimize(objective, space, n_calls=50, random_state=1)

    print('Maximum accuracy: {:.4f}%'.format(-result.fun * 100))
    print('Optimal weights: {}'.format(result.x))

    final_pred = np.zeros_like(r[0])

    for i in range(len(result.x)):
        final_pred += r[i] * result.x[i]
    
    np.save('ensemble_score.npy', final_pred)
