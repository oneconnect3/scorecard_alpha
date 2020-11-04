import logging
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

def main():
    logger = logging.getLogger(__name__)

    #1 数据读取：训练集数据、测试集数据
    logger.info('1. reading data:')

    train_data = pd.read_csv('data/raw/cs-training.csv', index_col=0)
    test_data = pd.read_csv('data/raw/cs-test.csv', index_col=0)
    logger.info('train_data.head:\n{}'.format(train_data.head()))

    #2 数据探索分析

    #3 数据预处理
    logger.info('3 preprocessing:')
   
    #3.1 异常值处理
    logger.info('3.1 removing outliers:')

    def remove_outliers(data):
        data=data[data['RevolvingUtilizationOfUnsecuredLines']<1]
        data=data[data['age']>18]
        data=data[data['NumberOfTime30-59DaysPastDueNotWorse']<80]
        data=data[data['NumberOfTime60-89DaysPastDueNotWorse']<80]
        data=data[data['NumberOfTimes90DaysLate']<80]
        data=data[data['NumberRealEstateLoansOrLines']<50]
        return data
        
    train_data=remove_outliers(train_data)
    test_data=remove_outliers(test_data)

    #查看经过异常值处理后是否还存在异常值
    # train_data.loc[(train_data['RevolvingUtilizationOfUnsecuredLines']>1)|(train_data['age']<18)|(train_data['NumberOfTime30-59DaysPastDueNotWorse']>80)|(train_data['NumberOfTime60-89DaysPastDueNotWorse']>80)|(train_data['NumberOfTimes90DaysLate']>80)|(train_data['NumberRealEstateLoansOrLines']>50)]
    # test_data.loc[(test_data['RevolvingUtilizationOfUnsecuredLines']>1)|(test_data['age']<18)|(test_data['NumberOfTime30-59DaysPastDueNotWorse']>80)|(test_data['NumberOfTime60-89DaysPastDueNotWorse']>80)|(test_data['NumberOfTimes90DaysLate']>80)|(test_data['NumberRealEstateLoansOrLines']>50)]
    print(train_data.shape)
    print(test_data.shape)

    # 缺失值处理
    logging.info('3.2 missing values:')
    # 对家属数量的缺失值进行删除
    logging.info('3.2.1 filling value for %s' % 'NumberOfDependents')
    train_data=train_data[train_data['NumberOfDependents'].notnull()]
    print(train_data.shape)
    test_data=test_data[test_data['NumberOfDependents'].notnull()]
    print(test_data.shape)

    # 对月收入缺失值用随机森林的方法进行填充--训练集
    logging.info('3.2.2 filling value for %s' % 'MonthlyIncome')

    # 创建随机森林函数
    def fillmonthlyincome(data):
        from sklearn.ensemble import RandomForestRegressor
        known = data[data['MonthlyIncome'].notnull()]
        unknown = data[data['MonthlyIncome'].isnull()]
        x_train = known.iloc[:,[1,2,3,4,6,7,8,9,10]]
        y_train = known.iloc[:,5]
        x_test = unknown.iloc[:,[1,2,3,4,6,7,8,9,10]]
        rfr = RandomForestRegressor(random_state=0,n_estimators=200,max_depth=3,n_jobs=-1)
        pred_y = rfr.fit(x_train,y_train).predict(x_test)
        return pred_y

    # 用随机森林填充训练集缺失值
    predict_data=fillmonthlyincome(train_data)
    train_data.loc[train_data['MonthlyIncome'].isnull(),'MonthlyIncome']=predict_data
    print(train_data.info())

    # 缺失值和异常值处理完后进行检查
    print(train_data.isnull().sum())
    print(test_data.isnull().sum())


    # 建立共线性表格
    logging.info('3.3 handle correlation')
        
    correlation_table = pd.DataFrame(train_data.corr())
    print(correlation_table)

    train_data.to_csv('data/interim/cs-training.csv')
    test_data.to_csv('data/interim/cs-test.csv')

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
