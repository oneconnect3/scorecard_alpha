## 评分卡模型


### 特征说明
y:SeriousDlqin2yrs：好坏客户, 是否发生90天以上逾期

x1:RevolvingUtilization Of UnsecuredLines：无担保放款的循环利用：除了不动产和像车贷那样除以信用额度总和的无分期付款债务的信用卡和个人信用额度总额/可用额度比值

x2:Age：借款人年龄

x3:NumberOfTime30-59DaysPastDueNotWorse：30-59天逾期次数

x4:DebtRatio：负债比例

x5:MonthlyIncome：月收入

x6:Number Of OpenCreditLinesAndLoans：开放式信贷和贷款数量

x7:NumberOfTimes90DaysLate：90天逾期次数：借款者有90天或更高逾期的次数

x8:NumberReal Estate Loans Or Lines：不动产贷款或额度数量：抵押贷款和不动产放款包括房屋净值信贷额度

x9:Number Of Time 60-89Days PastDue Not Worse：60-89天逾期次数

x10:NumberOfDependents：家属数量,不包括本人在内的家属数量



### 运行

1. 数据预处理

```bash
#.
(scorecard) $ python src/data/make_dataset.py
```

2. 特征工程

 ```bash
#.
(scorecard) $ python src/features/build_features.py
 ```

3. 建立模型

 ```bash
#.
(scorecard) $ python src/model/train_model.py
 ```



###  项目目录

```bash
.
├── README.md
├── data
│   ├── external
│   │   └── Data\ Dictionary.xls
│   ├── interim
│   ├── processed
│   └── raw
│       ├── cs-test.csv
│       └── cs-training.csv
├── notebooks
│   ├── build_scorecard_model.ipynb
│   └── build_scorecard_model_from_scratch.ipynb
├── requirements.txt
└── src
    ├── data
    │   └── make_dataset.py
    ├── features
    │   └── build_features.py
    ├── model
    │   └── train_model.py
    └── visualization
        └── analyze_dataset.py
```



### 参考

1. https://cloud.tencent.com/developer/article/1448182
2. https://www.kesci.com/mw/project/5d57b251c143cf002b239c29/dataset
3. https://www.kaggle.com/c/GiveMeSomeCredit/data
