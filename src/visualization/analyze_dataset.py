import logging
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

def main():
    logger = logging.getLogger(__name__)

    #1）数据读取：训练集数据、测试集数据
    logger.info('1. reading data:')

    train_data = pd.read_csv('data/raw/cs-training.csv', index_col=0)
    test_data= pd.read_csv('data/raw/cs-test.csv', index_col=0)

    print('-------------------')
    print("train_data.shape:\n{}".format(train_data.shape))
    print(train_data.info())
    print("test_data.shape:\n{}".format(test_data.shape))
    print(test_data.info())
    print('-------------------')

    logger.info('2. analyzing data:')
    print('-------------------')

    # 好坏客户分布: 样本标签是否均衡, 若否, 后面需要使用balance参数
    logger.info('2.1 data distribution for %s' % 'SeriousDlqin2yrs')
    badnum=train_data['SeriousDlqin2yrs'].sum()
    goodnum=train_data['SeriousDlqin2yrs'].count()-train_data['SeriousDlqin2yrs'].sum()
    print('训练集数据中，好客户数量为：%i,坏客户数量为：%i,坏客户所占比例为：%.2f%%' %(goodnum,badnum,(badnum/train_data['SeriousDlqin2yrs'].count())*100))

    # 可用额度比值特征分布: 异常值检查, 比值应当小于1
    logger.info('2.2 data distribution for %s' % 'RevolvingUtilizationOfUnsecuredLines')
    print(train_data['RevolvingUtilizationOfUnsecuredLines'].describe())

    # 年龄分布: 极值处理
    logger.info('2.3 data distribution for %s' % 'age')
    print(train_data['age'].describe())
    # train_data[train_data['age']<18]#只有1条，而且年龄为0，后面当做异常值删除
    # train_data[train_data['age']>100]#较多且连续，可暂时保留

    # 逾期30-59天 | 60-89天 | 90天笔数分布
    logger.info('2.4 data distribution for %s %s %s' % ('NumberOfTime30-59DaysPastDueNotWorse','NumberOfTime60-89DaysPastDueNotWorse','NumberOfTimes90DaysLate'))
    print(train_data[train_data['NumberOfTime30-59DaysPastDueNotWorse']>13].count())
    #这里可以看出逾期30-59天次数大于13次的有269条，大于80次的也是269条，说明这些是异常值，应该删除
    print(train_data[train_data['NumberOfTime60-89DaysPastDueNotWorse']>13].count())
    #这里可以看出逾期60-89天次数大于13次的有269条，大于80次的也是269条，说明这些是异常值，应该删除
    print(train_data[train_data['NumberOfTimes90DaysLate']>17].count())
    #这里可以看出逾期90天以上次数大于17次的有269条，大于80次的也是269条，说明这些是异常值，应该删除

    # 负债率特征分布
    logger.info('2.5 data distribution for %s' % 'DebtRatio')
    print(train_data['DebtRatio'].describe())
    print(train_data[train_data['DebtRatio']>1].count())
    #因为大于1的有三万多笔，所以猜测可能不是异常值

    # 信贷数量特征分布
    logger.info('2.6 data distribution for %s' % 'NumberOfOpenCreditLinesAndLoans')
    print(train_data['NumberOfOpenCreditLinesAndLoans'].describe())
    #由于箱型图的上界值挺连续，所以可能不是异常值

    # 固定资产贷款数量
    logger.info('2.7 data distribution for %s' % 'NumberRealEstateLoansOrLines')
    print(train_data['NumberRealEstateLoansOrLines'].describe())
    #查看箱型图发现最上方有异常值
    print(train_data[train_data['NumberRealEstateLoansOrLines']>32].count())

    # 家属数量分布
    logger.info('2.8 data distribution for %s' % 'NumberOfDependents')

    print(train_data['NumberOfDependents'].describe())
    print(train_data[train_data['NumberOfDependents']>15].count())
    
    # 月收入分布
    logger.info('2.9 data distribution for %s' % 'MonthlyIncome')
    print(train_data['MonthlyIncome'].describe())
    print(train_data[train_data['MonthlyIncome']>2000000].count())

    #查看缺失比例
    x=(test_data['age'].count()-test_data['MonthlyIncome'].count())/test_data['age'].count()
    print('月收入缺失数量比例为%.2f%%' % (x*100))
    #由于月收入缺失数量过大，后面采用随机森林的方法填充缺失值

    print('-------------------')



if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
