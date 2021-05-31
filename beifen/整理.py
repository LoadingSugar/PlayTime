from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
cv_scores = cross_val_score(clf, X_train, y_train, scoring='roc_auc')
print("Mean Cross-validation scores: {}".format(cv_scores.mean()))
#===
from feature_engine.categorical_encoders import OneHotCategoricalEncoder
def train_test_split_ohe(data):
    X = data.drop('Attrition_rate', axis = 1)
    y = data.Attrition_rate.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 1/3, random_state=0)

    ohe = OneHotCategoricalEncoder(drop_last=True)
    X_train = ohe.fit_transform(X_train).values
    X_test = ohe.transform(X_test).values

    return X_train, X_test, y_train, y_test

def model_training_pred_results(X_train, X_test, y_train, y_test):
  print('LinearRegression')
  regressor = LinearRegression()
  regressor.fit(X_train, y_train)    
  y_pred = regressor.predict(X_test)
  print('MSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
  print('MAE:', mean_absolute_error(y_test, y_pred))
  print('MAPE:', mean_absolute_percentage_error(y_test, y_pred))

X_train, X_test, y_train, y_test = train_test_split_ohe(data)
model_training_pred_results(X_train, X_test, y_train, y_test)

from sklearn.feature_selection import RFE
logreg = LogisticRegression()
rfe = RFE(logreg, 15)             # running RFE with 13 variables as output
rfe = rfe.fit(X_train, y_train)
#rfe.support_    #array([ True, False,  True,  True,  True,  True, False, False,  True,True,  True, False,  True, False,  True,  True,  True, False,False, False,  True,  True,  True])
list(zip(X_train.columns, rfe.support_, rfe.ranking_))
col = X_train.columns[rfe.support_]
X_train[col]



from sklearn.linear_model import LogisticRegressionCV
log_clf = LogisticRegressionCV(cv = 5)
log_clf.fit( X_train, train_labels )
train_test_inner['Pred1'] = log_clf.predict_proba(X_train)[:,1]
train_test_missing['Pred1'] = log_clf.predict_proba(X_test)[:,1]


y_pred=np.where(y_pred>0.5,1,0)



#杂项
    import random
    random.shuffle(li)

    x if (x>y) else y
    #集合
    并集：s.union(t) 或者 s | t
    交集：s.intersection(t) 或者 s & t
    差集：s.difference(t) 或者 s - t
    #采样
    train_data=data_model.sample(n=200,random_state=123)
    train_data=datamodel.sample(frac=0.7,random_state=123)
    testdata=data_model[~data_model.index.isin(train_data.index)]#剩下数据做测试

    y_preditc=list(map(lambda x: 1 if x>0.5 else 0, y_preditc))

    train.loc[ train['Season'].isin(years), : ]
    train.loc[ train['Season'].isin(years), ['as','as1']]
# EDA
        df_train.describe()
        df_test.info()
    ## 判断数据空
        all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
        all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
        missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

    df4 = df4.dropna(subset=['age', 'body','home.dest'])

    train_df.isnull().sum()
# 数据偏差
    ##画图
        from scipy import stats
        from scipy.stats import norm, skew #for some statistics
        import seaborn as sns
        color = sns.color_palette()
        sns.set_style('darkgrid')
        sns.distplot(train['SalePrice'] , fit=norm)
        fig = plt.figure()
        res = stats.probplot(train['SalePrice'], plot=plt)
        plt.show()
    ##计算偏差
        (mu, sigma) = norm.fit(train['SalePrice'])
        print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
                    #### 右偏差  log一下
        train["SalePrice"] = np.log1p(train["SalePrice"]) #log(1+x)

                    #####skew偏差，右长右偏 众数<中位数<均值  >0  对数变换，较小数据之间的差异将会变大(因为对数函数的斜率很小)，而较大数据之间的差异将减少(因为该分布中较大数据的斜率很小)    boxcox1p对数变换（λ=0），平方根变换（λ=1/2）和倒数变换（λ＝-１）
    ##对非类别数据进行偏差调整
        numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
        skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
        skewness = pd.DataFrame({'Skew' :skewed_feats})
        skewness.head(10)
    ####调整
        skewness = skewness[abs(skewness) > 0.75]
        from scipy.special import boxcox1p
        skewed_features = skewness.index
        for feat in skewed_features:
            #all_data[feat] += 1
            all_data[feat] = boxcox1p(all_data[feat], lam = 0.15)
        #all_data[skewed_features] = np.log1p(all_data[skewed_features])



#========================================================================================================================================================================================================================================================

# groupby
        features = features.groupby(['user_id', 'item_id']).behavior_type.agg({'behavior_types': 'sum'}).reset_index()

        item_Top50['item_id'] = item_Top50['item_id'].astype('str')
        item_Top50['item_id'] = ',' + item_Top50['item_id']
        item_Top50=item_Top50.groupby(['user_id']).item_id.sum().reset_index()##sum可以对字符串进行拼接
        item_Top50['item_id'] = item_Top50['item_id'].apply(lambda x: x[1:]).drop_duplicates()
        item_Top50 = df_user_test.merge(item_Top50, on='user_id', how='left')[['user_id', 'item_id']].groupby(['user_id']).head(50)

        df_train.loc[df_train.Age.isnull(), 'Age'] = df_train.groupby(['Sex','Pclass','Title']).Age.transform('median')

    ## 每个人前50
        item_Top50 = item_Top50.sort_values(by=['user_id', 'behavior_type'],ascending=False).groupby(['user_id']).head(50)


    ## 每个人不定长行数转换成list或者字符串
        train_test.groupby(['item_id'])['user_id'].agg(lambda x:','.join(list(x))).reset_index()
        data_['user_id']=list(data_['user_id'].map(lambda x:x.split(',')))
        #一步到位
        train_test.groupby(['item_id'])['user_id'].agg(list).reset_index()

# 清理内存
        gc.collect()

# 保存 不存行列名称
        item_Top50[['user_id', 'item_id']].to_csv('result/submit.csv', sep='\t',  header=None, index=False)

# 重建DateFrame
        train2.reset_index(drop=True,inplace=True)
        item_Top50 = df_item_selected.sort_values(by=['itemCount'],ascending=False).head(50)['item_id'].reset_index(drop=True)
# 热门商品top50 得分从 -50 到 -1，行为top 50 取得分大于零的，这样合并后效果就是热门商品作为补齐项
        item_Top50 = pd.DataFrame({
            'behavior_type': [i for i in range(-1, -51, -1)],
            'item_id': item_Top50
        })


# 去重填充
        processed_df['Embarked'].fillna('C', inplace=True)

        train_data = train_df.copy()
        train_data["Age"].fillna(train_df["Age"].median(skipna=True), inplace=True)
        train_data["Embarked"].fillna(train_df['Embarked'].value_counts().idxmax(), inplace=True)

        data.drop_duplicates(subset='B',keep='first',inplace=True)
        df.interpolate(method='linear', limit_direction='forward', axis=0)#插值方式  nearest：最邻近插值法 zero：阶梯插值 slinear、linear：线性插值 quadratic、cubic：2、3阶B样条曲线插值

        processed_df['Age'] = processed_df.groupby(['Pclass','Sex','Parch','SibSp'])['Age'].transform(lambda x: x.fillna(x.mean()))
        processed_df['Age'] = processed_df.groupby(['Pclass','Sex','Parch'])['Age'].transform(lambda x: x.fillna(x.mean()))

        rankings = rankings.set_index(['rank_date']).groupby(['country_full'],group_keys = False).resample('D').first().fillna(method='ffill').reset_index()#按照country_full 填充缺失的天数数据

# 正则表达
        df_train['Title'] = df_train.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))#匹配括号第一个  re.search("([0-9]*)([a-z]*)([0-9]*)",a).group(0)整体取出

# 更换
        features.loc[features['behavior_type']=='cart','behavior_type'] = -.5
        processed_df['FamillySize'][processed_df['FamillySize'].between(1, 5, inclusive=False)] = 2
        processed_df['FamillySize'][processed_df['FamillySize']>5] = 3

        interval = (0, 5, 12, 18, 25, 35, 60, 120) 
        cats = ['babies', 'Children', 'Teen', 'Student', 'Young', 'Adult', 'Senior']
        df_train["Age_cat"] = pd.cut(df_train.Age, interval, labels=cats)

        processed_df['Title'] = processed_df['Title'].replace(['Mlle', 'Ms'], 'Miss')
        processed_df['Title'] = processed_df['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5})
        processed_df['Cabin'] = processed_df['Cabin'].map({cabin: p for p, cabin in enumerate(set(cab for cab in processed_df['Cabin']))})

        processed_df['IsAlone'] = np.where(processed_df['FamillySize']!=1, 0, 1)

        def tenure_lab(telcom) :
            if telcom["tenure"] <= 12 :
                return "Tenure_0-12"
            elif (telcom["tenure"] > 12) & (telcom["tenure"] <= 24 ):
                return "Tenure_12-24"
        telcom["tenure_group"] = telcom.apply(lambda telcom:tenure_lab(telcom), axis=1)

        train_['zheng']=train_.apply(lambda x:str(str(x[0])+'_'+str(x[1])), axis=1)

        temp['target'] = temp[['item_id', 'target_SKU']].apply(lambda x: 1 if x['item_id'] == x['target_SKU'] else 0, axis=1)#横向
#========================================================================================================================================================================================================================================================
# 规范化
    ## 类别数据筛选
        from sklearn.preprocessing import OneHotEncoder,LabelEncoder
        oh=OneHotEncoder(dtype='uint8')
        x=oh.fit_transform(train_data[cols])

        categorical_cols = telcom.select_dtypes(include='object').columns
        categorical_cols = [col for col in categorical_cols if col not in target_col + ignored_cols]

        numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index


        df_train = pd.get_dummies(df_train, columns=["Sex","Embarked","Age_cat","Fare_cat","Title"],prefix=["Sex","Emb","Age","Fare","Prefix"], drop_first=True)#==from sklearn.preprocessing import OneHotEncoder
                    pd.get_dummies( train_test_merge['Season'].astype(object) )

        le = LabelEncoder()#仅仅变成数字  
        for i in bin_cols :    data[i] = le.fit_transform(data[i])

        # 定义一个有序的字段和索引的映射函数
        def build_map(df, col_name):
            key = sorted(df[col_name].unique().tolist())
            m  = dict(zip(key , range(len(key))))
            df[col_name] = df[col_name].map(lambda x: m[x])
            return m, ke
        asin_map, asin_key = build_map(meta_df,"asin")

    ## 归一化
        #from sklearn.preprocessing import StandardScaler
        #sc = StandardScaler()
        #X_train = sc.fit_transform(X_train)
        #X_test = sc.fit_transform(X_test)

        #sc = StandardScaler()
        #X = pd.DataFrame(sc.fit_transform(X.values), index=X.index, columns=X.columns)
        #标准化                  如果数据集中含有噪声和异常值，可以选择标准化，标准化更加适合嘈杂的大数据集。
        from sklearn import preprocessing
        min_max_scaler = preprocessing.MinMaxScaler()
        X_minMax = min_max_scaler.fit_transform(temp)
        #归一化                 如果数据集小而稳定，可以选择归一化
        from sklearn.preprocessing import StandardScaler 
        source_data['New_Amount'] =  StandardScaler().fit_transform(source_data['Amount'].values.reshape(-1,1))
        # StandardScaler参数只接受矩阵
        data = source_data.drop(columns=['Amount','Time'])
#========================================================================================================================================================================================================================================================
# 划分label
        train = df_train.drop(["Survived","PassengerId"],axis=1)
        train_ = df_train["Survived"]

# 拼接
        pd.merge(df1,df3,on='name',how='left')
        matches = matches.merge(rankings, left_on=['date', 'away_team'],right_on=['rank_date', 'country_full'],suffixes=('_home', '_away'))#相同字段都有的，suffixes用于追加到重叠列名的末尾
    ## 在某些DataFrame出现过的
        meta_df = meta_df[meta_df['asin'].isin(review_df['asin'].unique())]
        meta_df = meta_df.reset_index(drop=True)
# train test
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        ntrain = train.shape[0]
        y_train = train.SalePrice.values
        all_data = pd.concat((train, test)).reset_index(drop=True)
        train = all_data[:ntrain]
        test = all_data[ntrain:]




#========================================================================================================================================================================================================================================================
# features
        numeric_cols = ['MonthlyCharges', 'TotalCharges', 'tenure']
        categorical_cols = [col for col in categorical_cols if col not in target_col + ignored_cols]
        for col in categorical_cols:    telcom[col] = LabelEncoder().fit_transform(telcom[col])
        telcom[numeric_cols] = StandardScaler().fit_transform(telcom[numeric_cols])

        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA, TruncatedSVD, FastICA
        from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
        data2 = StandardScaler().fit_transform(PCA(n_components=40, random_state=4).fit_transform(data[cols]))

        fica = FastICA(n_components=3)
        _X = fica.fit_transform(telcom[numeric_cols + categorical_cols])
        fica_data = pd.DataFrame(_X, columns=["FICA1", "FICA2", "FICA3"])
        telcom[["FICA1", "FICA2", "FICA3"]] = fica_data

        tsvd = TruncatedSVD(n_components=3)
        _X = tsvd.fit_transform(telcom[numeric_cols + categorical_cols])
        tsvd_data = pd.DataFrame(_X, columns=["TSVD1", "TSVD2", "TSVD3"])
        telcom[["TSVD1", "TSVD2", "TSVD3"]] = tsvd_data

        grp = GaussianRandomProjection(n_components=3)
        _X = grp.fit_transform(telcom[numeric_cols + categorical_cols])
        grp_data = pd.DataFrame(_X, columns=["GRP1", "GRP2", "GRP3"])
        telcom[["GRP1", "GRP2", "GRP3"]] = grp_data

        srp = SparseRandomProjection(n_components=3)
        _X = srp.fit_transform(telcom[numeric_cols + categorical_cols])
        srp_data = pd.DataFrame(_X, columns=["SRP1", "SRP2", "SRP3"])
        telcom[["SRP1", "SRP2", "SRP3"]] = srp_data

        numeric_cols.extend(pca_data.columns.values)


#========================================================================================================================================================================================================================================================
#label  采样

        from imblearn.over_sampling import SMOTE
        # 定义SMOTE模型，random_state相当于随机数种子的作用
        smote = SMOTE(sampling_strategy='minority', random_state=42)#对稀缺的类别进行重新采样
        os_smote_X, os_smote_Y = smote.fit_sample(train_df[numeric_cols + categorical_cols], train_df[target_col].values.ravel())
        train_df = pd.DataFrame(os_smote_X, columns=numeric_cols + categorical_cols)
        train_df['Churn'] = os_smote_Y#直接五五开

        from collections import Counter# Counter({0: 900, 1: 100})
        SMOTE(ratio = {2:500,3:1000},random_state = 42)
        X_smo, y_smo = smo.fit_sample(X, y)
        print(Counter(y_smo))






#========================================================================================================================================================================================================================================================

# StratifiedKFold+class
        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(n_splits=5, random_state=42)
        for train_index, test_index in skf.split(train2, train2['target']):
            clf = NuSVC(probability=True, kernel='poly', degree=4, gamma='auto', random_state=4, nu=0.59, coef0=0.053)
            clf.fit(train3[train_index,:],train2.loc[train_index]['target'])
            oof[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]
            preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits
            
            clf = neighbors.KNeighborsClassifier(n_neighbors=17, p=2.9)
            clf.fit(train3[train_index,:],train2.loc[train_index]['target'])
            oof_2[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]
            preds_2[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits
        print(roc_auc_score(train['target'], oof))
        print(roc_auc_score(train['target'], oof_2))
        print(roc_auc_score(train['target'], 0.8*oof+0.2*oof_2))
        print(roc_auc_score(train['target'], 0.95*oof+0.05*oof_2))
        print(roc_auc_score(train['target'], 1.05*oof-0.05*oof_2))

        sub['target'] = preds           sub.to_csv('submission.csv', index=False)
        sub['target'] = 0.8*preds+0.2*preds_2      sub.to_csv('submission_2.csv', index=False)
        sub['target'] = 0.95*preds+0.05*preds_2     sub.to_csv('submission_3.csv', index=False)
        sub['target'] = 1.05*preds-0.05*preds_2


#stack+Regressor
        from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
        from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import RobustScaler
        from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
        from sklearn.model_selection import KFold, cross_val_score, train_test_split
        from sklearn.metrics import mean_squared_error
        import xgboost as xgb
        import lightgbm as lgb
        n_folds = 5
        def rmsle_cv(model):
            kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
            rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))   #scoring==  Classification=(accuracy f1_micro roc_auc)   Regression=(max_error)
            return(rmse)

        lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
        ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
        KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
        GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                        max_depth=4, max_features='sqrt',
                                        min_samples_leaf=15, min_samples_split=10, 
                                        loss='huber', random_state =5)
        model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                                    learning_rate=0.05, max_depth=3, 
                                    min_child_weight=1.7817, n_estimators=2200,
                                    reg_alpha=0.4640, reg_lambda=0.8571,
                                    subsample=0.5213, silent=1,
                                    random_state =7, nthread = -1)
        model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                                    learning_rate=0.05, n_estimators=720,
                                    max_bin = 55, bagging_fraction = 0.8,
                                    bagging_freq = 5, feature_fraction = 0.2319,
                                    feature_fraction_seed=9, bagging_seed=9,
                                    min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
        score = rmsle_cv(lasso)       print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
        score = rmsle_cv(ENet)        print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
        score = rmsle_cv(KRR)         print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
        score = rmsle_cv(GBoost)      print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
        score = rmsle_cv(model_xgb)   print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
        score = rmsle_cv(model_lgb)   print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
    ###stack_simple
        #####改写加上shuffle
        class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
            def __init__(self, models):
                self.models = models
                
            # we define clones of the original models to fit the data in
            def fit(self, X, y):
                self.models_ = [clone(x) for x in self.models]
                
                # Train cloned base models
                for model in self.models_:
                    model.fit(X, y)

                return self
            
            #Now we do the predictions for cloned models and average them
            def predict(self, X):
                predictions = np.column_stack([
                    model.predict(X) for model in self.models_
                ])
                return np.mean(predictions, axis=1)  


        averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))
        score = rmsle_cv(averaged_models)
        print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    ###stack_plus
        class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
            def __init__(self, base_models, meta_model, n_folds=5):
                self.base_models = base_models
                self.meta_model = meta_model
                self.n_folds = n_folds
        
            # We again fit the data on clones of the original models
            def fit(self, X, y):
                self.base_models_ = [list() for x in self.base_models]
                self.meta_model_ = clone(self.meta_model)
                kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
                
                # Train cloned base models then create out-of-fold predictions
                # that are needed to train the cloned meta-model
                out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
                for i, model in enumerate(self.base_models):
                    for train_index, holdout_index in kfold.split(X, y):
                        instance = clone(model)
                        self.base_models_[i].append(instance)
                        instance.fit(X[train_index], y[train_index])
                        y_pred = instance.predict(X[holdout_index])
                        out_of_fold_predictions[holdout_index, i] = y_pred
                        
                # Now train the cloned  meta-model using the out-of-fold predictions as new feature
                self.meta_model_.fit(out_of_fold_predictions, y)
                return self
        
            #Do the predictions of all base models on the test data and use the averaged predictions as 
            #meta-features for the final prediction which is done by the meta-model
            def predict(self, X):
                meta_features = np.column_stack([
                    np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
                    for base_models in self.base_models_ ])
                return self.meta_model_.predict(meta_features)

        stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),model = lasso)
        score = rmsle_cv(stacked_averaged_models)
        print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))


#Ensembling 
        def rmsle(y, y_pred):
            return np.sqrt(mean_squared_error(y, y_pred))

        stacked_averaged_models.fit(train.values, y_train)
        stacked_train_pred = stacked_averaged_models.predict(train.values)
        stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))
        print(rmsle(y_train, stacked_train_pred))

        model_xgb.fit(train, y_train)
        xgb_train_pred = model_xgb.predict(train)
        xgb_pred = np.expm1(model_xgb.predict(test))
        print(rmsle(y_train, xgb_train_pred))

        model_lgb.fit(train, y_train)
        lgb_train_pred = model_lgb.predict(train)
        lgb_pred = np.expm1(model_lgb.predict(test.values))
        print(rmsle(y_train, lgb_train_pred))

        print('RMSLE score on train data:')
        print(rmsle(y_train,stacked_train_pred*0.70 +xgb_train_pred*0.15 + lgb_train_pred*0.15 ))
        ensemble = stacked_pred*0.70 + xgb_pred*0.15 + lgb_pred*0.15##结果
#========================================================================================================================================================================================================================================================

# 画图
    ## 类别计数
        plt.figure(figsize=(12,5))
        sns.countplot(x='Title', data=df_train, palette="hls")
        plt.xlabel("Title", fontsize=16) #seting the xtitle and size
        plt.ylabel("Count", fontsize=16) # Seting the ytitle and size
        plt.title("Title Name Count", fontsize=20) 
        plt.xticks(rotation=45)
        plt.show()
    ## 交叉计数
        sns.countplot(x='Title', data=df_train, palette="hls",hue="Survived")
    ## 线拟合
        sns.distplot(age_high_zero_died["Age"], bins=24, color='r')
    ## 点点图
        sns.swarmplot(x='Age_cat',y="Fare",data=df_train,hue="Survived", palette="hls", )
    ## 热力特征相关图
        sns.heatmap(df_train.astype(float).corr(),vmax=1.0,  annot=True)
# 表
    ## 交叉表
        pd.crosstab(df_train.Age_cat, df_train.Survived)

        Age_fare = ['Pclass', 'Age_cat'] #seting the desired 
        cm = sns.light_palette("red", as_cmap=True)
        pd.crosstab(df_train[Age_fare[0]], df_train[Age_fare[1]],values=df_train['Fare'], aggfunc=['mean']).style.background_gradient(cmap = cm)



# 保存
    from datetime import timedelta, datetime
    df_sub.to_csv('submit-{}.csv'.format(datetime.now().strftime('%m%d_%H%M%S')), sep=',', index=False)

    with open('subA_lgb_{}.txt'.format(datetime.now().strftime('%m%d_%H%M%S')), 'w', encoding = 'utf-8') as file:
    file.write(sub)


# 时间
    import datetime
    import time
    start_time = datetime.datetime.now()
    time.sleep(5)
    end_time = datetime.datetime.now()
    delta = end_time - start_time
    delta_gmtime = time.gmtime(delta.total_seconds())
    duration_str = time.strftime("%H:%M:%S", delta_gmtime)
    print("start time {}".format(start_time))
    print("end time {}".format(end_time))
    print("delta_gmtime {}".format(delta_gmtime))
    print("duration_str {}".format(duration_str))
