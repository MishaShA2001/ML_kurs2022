# импортируем наш любимый pandas
import pandas as pd

# загружаем данные
data = pd.read_csv('train_dataset_train.csv', sep=',')
print(data.head())

# какие колонки нужно удалить
columns_to_drop = [
    'id',
    'ticket_id',
    'entrance_id',
    'station_id',
    'line_id'
]

# эту колонку тоже, возможно, надо удалить
columns_to_drop.append('entrance_nm')
data = data.drop(columns=columns_to_drop)
print('Ненужные колонки удалили.')

# время и день захода на станцию тоже как-то влияет
data['pass_dttm'] = pd.to_datetime(data['pass_dttm'])

# время суток
data['hour'] = data['pass_dttm'].dt.hour
data['time_of_day'] = data['hour'].apply(
    lambda hour: 'day' if hour>=10 and hour<=16 else (
                 'night' if hour>=0 and hour<=4 else (
                 'morning' if hour>=4 and hour<=9 else 'evening')))

# выходной?
data['weekday'] = data['pass_dttm'].dt.weekday
data['is_weekend'] = data['weekday'].apply(
    lambda weekday: 1 if weekday>4 else 0)
print('Даты и время преобразовали.')

# удаляем то, что уже не нужно
data = data.drop(columns=['pass_dttm', 'hour', 'weekday'])

# категориальные признаки
obj_feat = list(data.loc[:, data.dtypes=='object'].columns.values)
for feature in obj_feat:
    data[feature] = pd.Series(data[feature], dtype='category')
print('С категориальными признаками разобрались.')
    
from sklearn.model_selection import train_test_split

X = data.drop(columns=['time_to_under', 'label'])
y = data['time_to_under']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# далее разделим еще как бы
X_train, X_test, y_train, y_test = train_test_split(X_test, y_test, test_size=0.3)
print('Выборки получили.')

import mlflow

# папка назначения локального сервера
mlflow.set_tracking_uri()
print('С локальным сервером разобрались.')

import lightgbm as lgb
from sklearn.metrics import r2_score

# создаем эксперимент
mlflow.set_experiment('training_experiment_1')
experiment = mlflow.get_experiment_by_name('training_experiment_1')
with mlflow.start_run(experiment_id=experiment.experiment_id):
    # параметры модели
    n_estimators = 11
    num_leaves = 11

    gbm = lgb.LGBMRegressor(objective='regression',
                            n_estimators=n_estimators,
                            num_leaves=num_leaves)
    gbm.fit(X_train, y_train)
    y_predict = gbm.predict(X_test, num_iteration=gbm.best_iteration_)

    r2 = r2_score(y_test, y_predict)

    print('Начинаем записывать данные run.')
    mlflow.log_param('n_estimators', n_estimators)
    mlflow.log_param('num_leaves', num_leaves)
    mlflow.log_metric('r2', r2)
    mlflow.sklearn.log_model(gbm, 'model')
    mlflow.end_run()
    print('Закончили run.')
