import logging
import warnings
warnings.filterwarnings("ignore")
import numpy as np

import pandas as pd

def main(key_cols):
    logger = logging.getLogger(__name__)

    # 数据读取：训练集数据、测试集数据
    logger.info('reading data:')

    train_data = pd.read_csv('data/processed/cs-training.csv', index_col=0)
    test_data = pd.read_csv('data/processed/cs-test.csv', index_col=0)
    logger.info('train_data.head:\n{}'.format(train_data.head()))
    x1_name = 'RevolvingUtilizationOfUnsecuredLines'
    x2_name = 'age'
    x3_name = 'NumberOfTime30-59DaysPastDueNotWorse'
    x7_name = 'NumberOfTimes90DaysLate'
    x9_name = 'NumberOfTime60-89DaysPastDueNotWorse'

    Y = train_data['SeriousDlqin2yrs'] # 因变量
    X = train_data.drop(['SeriousDlqin2yrs','DebtRatio','MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans','NumberRealEstateLoansOrLines','NumberOfDependents'], axis=1)     # 自变量，剔除对因变量影响不明显的变量
    X = X.iloc[:,-5:]
    print('X.head:\n{}'.format(X.head()))
    
    # 5.2 STATSMODEL包来建立逻辑回归模型得到回归系数，后面可用于建立标准评分卡
    import statsmodels.api as sm
    X1 = sm.add_constant(X)
    logit = sm.Logit(Y,X1)
    result = logit.fit()
    print(result)
    print(result.summary2()) # statsmodels=0.8.0

    # 6 模型评估: 对训练集模型做模拟和评估
    print(train_data.head())

    test_X = X.copy()   # 测试数据特征
    test_Y = Y.copy()   # 测试数据标签
    from sklearn import metrics
    X3 = sm.add_constant(test_X)
    resu = result.predict(X3)                          # 进行预测
    fpr,tpr,threshold = metrics.roc_curve(test_Y,resu) # 评估算法
    rocauc = metrics.auc(fpr,tpr)                      # 计算AUC
    print('AUC: %0.2f' % rocauc)

    p = 20/np.log(2)                                   # 比例因子
    q = 600-20*np.log(20)/np.log(2)                    # 偏移量
    x_coe = round(result.params, 4)                    # 回归系数
    # x_coe=[-2.7340,0.6526,0.5201,0.5581,0.5943,0.4329] 

    baseScore=round(q+p*x_coe[0],0)
    # 个人总评分 = 基础分+各部分得分
    def get_score(coe,woe,factor):
        scores = []
        for w in woe:
            score = round(coe*w*factor,0)
            scores.append(score)
        return scores

    # 每一项得分
    x1_score = get_score(x_coe[1],train_data['RevolvingUtilizationOfUnsecuredLines_woe'],p)
    x2_score = get_score(x_coe[2],train_data['age_woe'],p)
    x3_score = get_score(x_coe[3],train_data['NumberOfTime30-59DaysPastDueNotWorse_woe'],p)
    x7_score = get_score(x_coe[4],train_data['NumberOfTimes90DaysLate_woe'],p)
    x9_score = get_score(x_coe[5],train_data['NumberOfTime60-89DaysPastDueNotWorse_woe'],p)

    def compute_score(series,cut,score):
        res = []
        i = 0
        while i < len(series):
            value = series.iloc[i]
            j = len(cut) - 2
            m = len(cut) - 2
            while j >= 0:
                if value >= cut[j]:
                    j = -1
                else:
                    j -= 1
                    m -= 1
            res.append(score[m])
            i += 1

        return res

    ninf = float('-inf') # 负无穷大
    pinf = float('inf')  # 正无穷大

    x1_cut =[ninf, 0.0295, 0.148, 0.5211, pinf] 
    x2_cut = [ninf, 34.0, 40.0, 45.0, 50.0, 54.0, 59.0, 64.0, 71.0, pinf]
    cutx3 = [ninf, 0, 1, 3, 5, pinf] 
    cutx7 = [ninf, 0, 1, 3, 5, pinf] 
    cutx9 = [ninf, 0, 1, 3, pinf]

    train_data['BaseScore'] = np.zeros(len(train_data))+baseScore
    train_data['x1'] = compute_score(train_data['RevolvingUtilizationOfUnsecuredLines'], x1_cut, x1_score)
    train_data['x2'] = compute_score(train_data['age'], x2_cut, x2_score)
    train_data['x3'] = compute_score(train_data['NumberOfTime30-59DaysPastDueNotWorse'], cutx3, x3_score)
    train_data['x7'] = compute_score(train_data['NumberOfTimes90DaysLate'], cutx7, x7_score)
    train_data['x9'] = compute_score(train_data['NumberOfTime60-89DaysPastDueNotWorse'], cutx9,x9_score)
    train_data['Score'] = train_data['x1'] + train_data['x2'] + train_data['x3'] + train_data['x7'] +train_data['x9']  + baseScore

    scoretable_tr = train_data.iloc[:,[0,-7,-6,-5,-4,-3,-2,-1]]  #选取需要的列，就是评分列
    scoretable_tr.head()

    colNameDict={'x1':'RevolvingUtilizationOfUnsecuredLines','x2':'age','x3':'NumberOfTime30-59DaysPastDueNotWorse',
                'x7':'NumberOfTimes90DaysLate','x9':'NumberOfTime60-89DaysPastDueNotWorse'}
    scoretable_tr = scoretable_tr.rename(columns=colNameDict, inplace=False)
    scoretable_tr.to_excel('data/processed/scorecard_tr.xlsx', index=False)

    # 8.对测试集进行预测和转化为信用评分卡
    # Logistic回归模型——将训练集的数据分为测试集和训练集
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=1024)

    clf = LogisticRegression()
    clf.fit(x_train, y_train)

    # 对测试集做预测
    score_proba = clf.predict_proba(x_test)
    y_predproba = score_proba[:,1]
    coe = clf.coef_
    print(coe)

    # 对模型做评估
    from sklearn.metrics import roc_curve,auc
    fpr,tpr,threshold = roc_curve(y_test,y_predproba)
    auc_score = auc(fpr,tpr)
    print(max(tpr-fpr))

    # 自变量，剔除对因变量影响不明显的变量
    test_data = test_data.drop(['DebtRatio','MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans','NumberRealEstateLoansOrLines','NumberOfDependents'],axis=1)
    X_test = test_data.iloc[:,-5:]
    Y_test = test_data['SeriousDlqin2yrs']   #因变量

    #对测试集做预测
    score_proba = clf.predict_proba(X_test)
    test_data['y_predproba'] = score_proba[:,1]
    coe = clf.coef_
    print('-------------------')
    print(coe)

    # 每一项得分
    x1_score = get_score(x_coe[1],test_data['RevolvingUtilizationOfUnsecuredLines_woe'],p)
    x2_score = get_score(x_coe[2],test_data['age_woe'],p)
    x3_score = get_score(x_coe[3],test_data['NumberOfTime30-59DaysPastDueNotWorse_woe'],p)
    x7_score = get_score(x_coe[4],test_data['NumberOfTimes90DaysLate_woe'],p)
    x9_score = get_score(x_coe[5],test_data['NumberOfTime60-89DaysPastDueNotWorse_woe'],p)

    test_data['BaseScore'] = np.zeros(len(test_data)) + baseScore
    test_data['x1'] = compute_score(test_data['RevolvingUtilizationOfUnsecuredLines'], x1_cut, x1_score)
    test_data['x2'] = compute_score(test_data['age'], x2_cut, x2_score)
    test_data['x3'] = compute_score(test_data['NumberOfTime30-59DaysPastDueNotWorse'], cutx3, x3_score)
    test_data['x7'] = compute_score(test_data['NumberOfTimes90DaysLate'], cutx7, x7_score)
    test_data['x9'] = compute_score(test_data['NumberOfTime60-89DaysPastDueNotWorse'], cutx9, x9_score)
    test_data['Score'] = test_data['x1'] + test_data['x2'] + test_data['x3'] + test_data['x7'] +test_data['x9']  + baseScore

    scoretable_te = test_data.iloc[:,[0,-8,-7,-6,-5,-4,-3,-2,-1]]  #选取需要的列，就是评分列
    print(scoretable_te.head())

    scoretable_te = scoretable_te.rename(columns=colNameDict,inplace=False)
    scoretable_te.to_excel('data/processed/scorecard_te.xlsx', index=False)    

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    key_cols = ['RevolvingUtilizationOfUnsecuredLines', 'age',
       'NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTimes90DaysLate',
       'NumberOfTime60-89DaysPastDueNotWorse']
    logging.info("key_cols:\n{}".format(key_cols))

    main(key_cols)
