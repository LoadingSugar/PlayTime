## 打乱并划分 
    df = df_user.sample(frac=1.0)  
    cut_idx = int(round(valid * df.shape[0]))
    df_train_0, df_train_1 = df.iloc[:cut_idx], df.iloc[cut_idx:]

## 日期
    data['trx_tm'] = pd.to_datetime(data['trx_tm'], format='%Y/%m/%d')  # 转变为8位时间字符
    data['trx_tm'] = data['trx_tm'].dt.strftime('%Y%m%d')
    data['month']=data['trx_tm'].apply(lambda x:x[4:6])
    month_value=data.month.unique().tolist()
    month_value.sort()

## unstack()  使用unstack函数，将data2的第二层列索引转变成行索引
    a1=data.groupby(['id','page_tm'])['page_no'].count().unstack().reset_index()

## 计数
    ## 平均
    aa=data.groupby(by=['id'],as_index=False)['page_no'].agg({'pageno_count':'count'})
    a2=data.groupby(by=['id'], as_index=False)['page_tm'].agg({'click_daynum': 'nunique'})
    aa['click_day_count']=aa['pageno_count']/aa['click_daynum']

## stack  
    df.set_index(['产品','类别']).stack().reset_index()#之后的月份上变成行上  列
    df.stack(level=0)

    df.unstack(level=0)#从最外面开始，索引0的，列转行
## melt
    df_melt = pd.melt(df, id_vars='月份', value_vars=['别克英朗', '丰田卡罗拉', '大众速腾', '本田思域'],var_name='车型',value_name='数量')#id_vars不需要转换的  value_vars:需要转换的列名 varname和valuename是自定义设置对应的列名。 
## pivot
    evaluation = test[['id','d','sold']]
    evaluation = pd.pivot(evaluation, index='id', columns='d', values='sold').reset_index()
    evaluation.columns=['id'] + ['F' + str(i + 1) for i in range(28)]
## 直接转置
d_cols = [c for c in train.columns if 'd_' in c] 
past_sales = train.set_index('id')[d_cols].T.merge(calender.set_index('d')['date'],
                                                   left_index=True,
                                                   right_index=True,
                                                   validate='1:1').set_index('date')

## transform
    sales['item_sold_avg'] = sales.groupby('item_id')['sold'].transform('mean').astype(np.float16)

    sales['expanding_sold_mean'] = sales.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])['sold'].transform(lambda x: x.rolling(window=6).mean()).astype(np.float16)#向后滚动6个做平均

    df['Open'].rolling(30).mean()#滚动30天，没有的用NAN填充
    df.shift(10)    #后移动  #time shifting其实就是把索引往前或者往后挪动

## 调参

    from lightgbm import LGBMRegressor as lgb
    from hyperopt import hp, tpe, fmin
    from sklearn.model_selection import cross_val_score

    valgrid = {'n_estimators':hp.quniform('n_estimators', 900, 1200, 100),
            'learning_rate':hp.quniform('learning_rate', 0.1, 0.4, 0.1),
            'max_depth':hp.quniform('max_depth', 4,8,1),
            'num_leaves':hp.quniform('num_leaves', 25,75,25),
            'subsample':hp.quniform('subsample', 0.5, 0.9, 0.1),
            'colsample_bytree':hp.quniform('colsample_bytree', 0.5, 0.9, 0.1),
            'min_child_weight':hp.quniform('min_child_weight', 200, 500, 100) 
            }

    def objective(params):
        params = {'n_estimators': int(params['n_estimators']),
                'learning_rate': params['learning_rate'],
                'max_depth': int(params['max_depth']),
                'num_leaves': int(params['num_leaves']),
                'subsample': params['subsample'],
                'colsample_bytree': params['colsample_bytree'],
                'min_child_weight': params['min_child_weight']}
        
        lgb_a = lgb(**params)
        score = cross_val_score(lgb_a, X_train, y_train, cv=2, n_jobs=-1).mean()
        return score

    bestP = fmin(fn= objective, space= valgrid, max_evals=20, rstate=np.random.RandomState(123), algo=tpe.suggest)




    
    model = lightgbm.LGBMRegressor(
            n_estimators = int(bestP['n_estimators']),
            learning_rate = bestP['learning_rate'],
            subsample = bestP['subsample'],
            colsample_bytree = bestP['colsample_bytree'],
            max_depth = int(bestP['max_depth']),
            num_leaves = int(bestP['num_leaves']),
            min_child_weight = int(bestP['min_child_weight']))

    print('Prediction for Store: {}**'.format(d_store_id[store]))
    model.fit(X_train, y_train, eval_set=[(X_train,y_train),(X_valid,y_valid)], eval_metric='rmse', verbose=20, early_stopping_rounds=20)
    validation_prediction[X_valid.index] = model.predict(X_valid)
    eval_prediction[X_test.index] = model.predict(X_test)
    filename = 'model'+str(d_store_id[store])+'.pkl'

## 降低内存，并且dataframe中的object类型转成
    import numpy as np
    def reduce_mem_usage(df):
    
        start_mem = df.memory_usage().sum() / 1024**2
        print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
            else:
                df[col] = df[col].astype('category')

        end_mem = df.memory_usage().sum() / 1024**2
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
        
        return df

    reduce_mem_usage(cal)

## 可以变回的类别编码
    d_id = dict(zip(sales.id.cat.codes, sales.id))
    cols = sales.dtypes.index.tolist()
    types = sales.dtypes.values.tolist()
    for i,type in enumerate(types):
        if type.name == 'category':
            sales[cols[i]] = sales[cols[i]].cat.codes#编码
    evaluation.id = evaluation.id.map(d_id)#转换回来