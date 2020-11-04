import logging
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pdb

import pandas as pd
from scipy.stats import spearmanr

def main():
    logger = logging.getLogger(__name__)
    # 数据读取：训练集数据、测试集数据
    logger.info('reading data:')

    train_data = pd.read_csv('data/interim/cs-training.csv', index_col=0)
    test_data = pd.read_csv('data/interim/cs-test.csv', index_col=0)

    logger.info('train_data.head:\n{}'.format(train_data.head()))

    # 4 特征选择
    logger.info('4. feature engineering:')
    # 4.1 分箱
    ## 4.1.1 连续性变量
    logger.info('4.1.1 monotonic-binning:')

    def compute_woe(bad_i, bad_t, good_t, n_i):
        good_i = n_i-bad_i
        return np.log((bad_i/bad_t)/(good_i/good_t))

    def transform_mono_bin(Y, X, n=10):
        """连续性变量: 定义自动分箱函数---最优分箱

        Args:
            Y ([type]): target变量
            X ([type]): 待分箱变量
            n (int, optional): n为分箱数量. Defaults to 10.

        Returns:
            [type]: d4, iv, cut, woe
        """
        
        r = 0                    # 设定斯皮尔曼相关系数初始值为0
        bad_t = Y.sum()          # 计算坏样本数
        good_t = Y.count()-bad_t # 计算好样本数
        
        # 下面这段就是分箱的核心: 机器来选择指定最优的分箱节点，代替我们自己来设置
        while np.abs(r) < 1:                                                # Fit the data for monotonic binning
            d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.qcut(X, n)})    # 用pd.qcut实现最优分箱，Bucket：将X分为n段，n由斯皮尔曼相关系数决定    
            d2 = d1.groupby('Bucket', as_index=True)                        # 按照分箱结果进行分组聚合        
            r, p = spearmanr(d2.mean().X, d2.mean().Y)                      # 以斯皮尔曼系数作为分箱终止条件
            n = n - 1    

        d3 = pd.DataFrame(d2.X.min(), columns=['min']) 
        d3['min'] = d2.min().X                          # 箱体的左边界
        d3['max'] = d2.max().X                          # 箱体的右边界
        d3['bad_i'] = d2.sum().Y                        # 每个箱体中坏样本的数量
        d3['n_i'] = d2.count().Y                        # 每个箱体的总样本数
        d3['rate'] = d2.mean().Y
        print('d3.rate:\n{}'.format(d3['rate']))
        print('----------------------')
        d3['woe_i']= compute_woe(d3['bad_i'], bad_t, good_t, d3['n_i']) # 计算每个箱体的woe值
        d3['p_bad'] = d3['bad_i']/bad_t                                 # 每个箱体中坏样本所占坏样本总数的比例
        d3['p_good'] = (d3['n_i']-d3['bad_i'])/good_t                   # 每个箱体中好样本所占好样本总数的比例
        iv = ((d3['p_bad']-d3['p_good'])*d3['woe_i']).sum()             # 计算变量的iv值
        d4 = (d3.sort_values(by='min')).reset_index(drop=True)          # 对箱体从大到小进行排序
        print('d4:\n{}'.format(d4))
        print('Information Value: {}'.format(iv))

        woe = list(d4['woe_i'].round(3))    
        cut = []                         # cut: 存放箱段节点
        cut.append(float('-inf'))        # 在列表前加-inf
        for i in range(1,n+1):           # n: 前面的分箱的分割数, 所以分成n+1份
            q_i = X.quantile(i/(n+1))    # q_i: quantile分位数, 得到分箱的节点
            cut.append(round(q_i,4))     # 保留4位小数, 并保存至cut
        cut.append(float('inf'))         # 在列表后加inf
        
        return d4, iv, cut, woe

    x1_d,x1_iv,x1_cut,x1_woe = transform_mono_bin(train_data['SeriousDlqin2yrs'],train_data.RevolvingUtilizationOfUnsecuredLines)
    x2_d,x2_iv,x2_cut,x2_woe = transform_mono_bin(train_data['SeriousDlqin2yrs'],train_data.age) 
    x4_d,x4_iv,x4_cut,x4_woe = transform_mono_bin(train_data['SeriousDlqin2yrs'],train_data.DebtRatio) 
    x5_d,x5_iv,x5_cut,x5_woe = transform_mono_bin(train_data['SeriousDlqin2yrs'],train_data.MonthlyIncome)

    ## 4.1.1 离散性变量--- 手动分箱
    logger.info('4.1.1 binning:')

    def custom_bin(Y, X, custom_cut):   
        """离散性变量: 手动分箱

        Args:
            Y ([type]): 各个特征变量
            X ([type]): 用户好坏标签
            custom_cut ([type]): 各个分箱

        Returns:
            [type]: d4, iv, woe
        """
        bad_t = Y.sum()             # 计算坏样本数
        good_t = Y.count()-bad_t    # 计算好样本数

        d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.cut(X, custom_cut)}) # 建立个数据框: pd.qcut vs. pd.cut: 后者更不均匀  
        d2 = d1.groupby('Bucket', as_index=True)                             # 按照分箱结果进行分组聚合
        d3 = pd.DataFrame(d2.X.min(), columns=['min'])                       # 添加min列, 不用管里面的d2.X.min()
        d3['min'] = d2.min().X    
        d3['max'] = d2.max().X    
        d3['bad_i'] = d2.sum().Y    
        d3['n_i'] = d2.count().Y    
        d3['rate'] = d2.mean().Y
        d3['woe_i']= compute_woe(d3['bad_i'], bad_t, good_t, d3['n_i']) # 计算每个箱体的woe值
        d3['p_bad'] = d3['bad_i']/bad_t                                 # 每个箱体中坏样本所占坏样本总数的比例
        d3['p_good'] = (d3['n_i'] - d3['bad_i'])/good_t                 # 每个箱体中好样本所占好样本总数的比例
        iv = ((d3['p_bad']-d3['p_good'])*d3['woe_i']).sum()             # 计算变量的iv值 
        d4 = (d3.sort_values(by='min')).reset_index(drop=True)          # 对箱体从大到小进行排序
        woe =list(d4['woe_i'].round(3))

        return d4, iv, woe

    ninf = float('-inf') # 负无穷大
    pinf = float('inf')  # 正无穷大
    cutx3 = [ninf, 0, 1, 3, 5, pinf]
    cutx6 = [ninf, 1, 2, 3, 5, pinf]
    cutx7 = [ninf, 0, 1, 3, 5, pinf]
    cutx8 = [ninf, 0, 1, 2, 3, pinf]
    cutx9 = [ninf, 0, 1, 3, pinf]
    cutx10 = [ninf, 0, 1, 2, 3, 5, pinf]
    dfx3, ivx3, woex3 = custom_bin(train_data.SeriousDlqin2yrs, train_data['NumberOfTime30-59DaysPastDueNotWorse'], cutx3)
    dfx6, ivx6, woex6= custom_bin(train_data.SeriousDlqin2yrs, train_data['NumberOfOpenCreditLinesAndLoans'], cutx6)
    dfx7, ivx7, woex7 = custom_bin(train_data.SeriousDlqin2yrs, train_data['NumberOfTimes90DaysLate'], cutx7)
    dfx8, ivx8, woex8 = custom_bin(train_data.SeriousDlqin2yrs, train_data['NumberRealEstateLoansOrLines'], cutx8)
    dfx9, ivx9, woex9 = custom_bin(train_data.SeriousDlqin2yrs, train_data['NumberOfTime60-89DaysPastDueNotWorse'], cutx9)
    dfx10, ivx10, woex10 = custom_bin(train_data.SeriousDlqin2yrs, train_data['NumberOfDependents'], cutx10)

    # 4.2 特征选择
    # 4.2.1 特征选择---相关系数矩阵
    correlation_table = pd.DataFrame(train_data.corr()) # 计算各变量的相关性系数
    print(correlation_table)
    
    # 4.2.2 IV值筛选: 通过IV值判断变量预测能力的标准是:小于 0.02: unpredictive；0.02 to 0.1: weak；0.1 to 0.3: medium； 0.3 to 0.5: strong
    iv_arr = np.array([x1_iv, x2_iv, ivx3, x4_iv, x5_iv, ivx6, ivx7, ivx8, ivx9, ivx10]) # 各变量IV
    
    key_cols = train_data.columns[1:][iv_arr>0.1]
    print("predictive variables:{}".format(key_cols))

    # 5.1 模型准备: 在建立模型之前，我们需要将筛选后的变量转换为WoE值，便于信用评分
    def transform_woe(var, var_name, woe, cut):
        """替换成woe函数"""
        woe_name = var_name+'_woe'
        for i in range(len(woe)):                                   # len(woe) 得到woe里: 有多少个数值
            if i==0:
                var.loc[(var[var_name]<=cut[i+1]),woe_name] = woe[i]  #将woe的值按cut分箱的下节点, 顺序赋值给var的woe_name列, 分箱的第一段
            elif (i>0) and (i<=len(woe)-2):
                var.loc[((var[var_name]>cut[i])&(var[var_name]<=cut[i+1])),woe_name] = woe[i] #    中间的分箱区间
            else:
                var.loc[(var[var_name]>cut[len(woe)-1]),woe_name]=woe[len(woe)-1]   # 大于最后一个分箱区间的上限值, 最后一个值是正无穷
        
        return var
    
    x1_name = 'RevolvingUtilizationOfUnsecuredLines'
    x2_name = 'age'
    x3_name = 'NumberOfTime30-59DaysPastDueNotWorse'
    x7_name = 'NumberOfTimes90DaysLate'
    x9_name = 'NumberOfTime60-89DaysPastDueNotWorse'
    
    train_data = transform_woe(train_data,x1_name,x1_woe,x1_cut)
    train_data = transform_woe(train_data,x2_name,x2_woe,x2_cut)
    train_data = transform_woe(train_data,x3_name,woex3,cutx3)
    train_data = transform_woe(train_data,x7_name,woex7,cutx7)
    train_data = transform_woe(train_data,x9_name,woex9,cutx9)
    
    print(x1_cut, x2_cut, cutx3, cutx7, cutx9)

    test_data = transform_woe(test_data,x1_name,x1_woe,x1_cut)
    test_data = transform_woe(test_data,x2_name,x2_woe,x2_cut)
    test_data = transform_woe(test_data,x3_name,woex3,cutx3)
    test_data = transform_woe(test_data,x7_name,woex7,cutx7)
    test_data = transform_woe(test_data,x9_name,woex9,cutx9)

    # from sklearn.model_selection import train_test_split
    
    # trainset, testset = train_test_split(train_data, random_state=1024, test_size=0.25)
    # trainset.to_csv('data/processed/cs-training-tr.csv')
    # testset.to_csv('data/processed/cs-training-te.csv')
    train_data.to_csv('data/processed/cs-training.csv')
    test_data.to_csv('data/processed/cs-test.csv')

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()