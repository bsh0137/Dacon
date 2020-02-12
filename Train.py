import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('C:/Users/customer/Desktop/Dacon/train.csv',index_col =0)
test = pd.read_csv('test.csv',index_col=0)
sample_submission = pd.read_csv('sample_submission.csv', index_col=0)

column_number = {}
for i, column in enumerate(sample_submission.columns):
    column_number[column] = i


def to_number(x, dic):
    return dic[x]

train['type_num'] = train['type'].apply(lambda x: to_number(x, column_number))

train_x = train.drop(columns=['type', 'type_num'], axis=1)
train_y = train['type_num']
test_x = test

forest = RandomForestClassifier(n_estimators=100, n_jobs=-1)
forest.fit(train_x, train_y)

y_pred = forest.predict_proba(test_x)

submission = pd.DataFrame(data=y_pred, columns=sample_submission.columns, index=sample_submission.index)
submission.to_csv('submission.csv', index=True)
