import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.metrics import mean_squared_error

import xgboost as xgb
# import lightgbm as lgb

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from scipy.stats import norm, skew
from scipy.special import boxcox1p

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

# find outliers
y_training_scaled = StandardScaler().fit_transform(train['SalePrice'][:, np.newaxis])
low = y_training_scaled[y_training_scaled[:, 0].argsort()][:10]
high = y_training_scaled[y_training_scaled[:, 0].argsort()][-10:]

data = pd.concat([train['SalePrice'], train['GrLivArea']], axis=1)
data.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0, 800000))
# plt.show()

# drop outliers
train = train.drop(train[(train['GrLivArea'] > 4500) & (train['SalePrice'] < 300000)].index)

# normalize data
train["SalePrice"] = np.log1p(train["SalePrice"])
y_train = train.SalePrice.values

num_train = train.shape[0]

num_test = test.shape[0]

combined = pd.concat([train, test])
combined.drop(['SalePrice'], axis=1, inplace=True)

print("all_data size is : {}".format(combined.shape))

# missing data
missing_total = combined.isnull().sum().sort_values(ascending=False)
missing_percentage = (combined.isnull().sum() / combined.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([missing_total, missing_percentage], axis=1, keys=['Total', 'Percent'])

# add values for missing data
# N/A means None
fields = ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageQual',
          'GarageFinish', 'GarageCond', 'GarageType', 'BsmtCond', 'BsmtExposure',
          'BsmtQual', 'BsmtFinType2', 'BsmtFinType1', 'MasVnrType', 'MSSubClass')

for f in fields:
    combined[f] = combined[f].fillna("None")

# N/A means 0
fields = ('GarageYrBlt', 'MasVnrArea', 'BsmtHalfBath', 'BsmtFullBath', 'BsmtUnfSF', 'BsmtFinSF2', 'BsmtFinSF1',
          'GarageArea', 'GarageCars', 'TotalBsmtSF')

for f in fields:
    combined[f] = combined[f].fillna(0)

# estimate the value to be average of each neighbor
combined["LotFrontage"] = combined.groupby("Neighborhood")["LotFrontage"].transform(lambda x : x.fillna(x.median()))

# estimate the values to be the mode
fields = ('MSZoning', 'Exterior1st', 'Exterior2nd', 'SaleType', 'KitchenQual', 'Electrical')

for f in fields:
    combined[f] = combined[f].fillna(combined[f].mode()[0])

# drop this irrelevant field
combined = combined.drop(['Utilities'], axis=1)

# N/A means Typ
combined["Functional"] = combined["Functional"].fillna("Typ")

# change some fields into categorical variable
combined['MSSubClass'] = combined['MSSubClass'].apply(str)
combined['OverallCond'] = combined['OverallCond'].apply(str)
combined['YrSold'] = combined['YrSold'].apply(str)
combined['MoSold'] = combined['MoSold'].apply(str)

# label the categorical fields with numbers
categories = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 'ExterQual',
              'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 'BsmtFinType2',
              'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope', 'LotShape',
              'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 'YrSold', 'MoSold')

for c in categories:
    lbe = LabelEncoder()
    combined[c] = lbe.fit_transform(list(combined[c].values))


# split the data and normalize skewed data
train = combined[:num_train]

numeric = train.dtypes[train.dtypes != "object"].index

skewed = train[numeric].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew': skewed})

skewed = skewness[abs(skewness) > 0.75].index

for s in skewed:
    train[s] = boxcox1p(train[s], 0.15)

# get dummy variable
combined = pd.get_dummies(combined)

train = combined[:num_train]
test = combined[num_train:]

# Cross Validation
n_fold = 10


def cv(model) :
    kf = KFold(n_splits=n_fold, random_state=48, shuffle=True).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return rmse


ridge = Ridge(alpha=9.0, fit_intercept=True)
lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0003, random_state=1, max_iter=50000))
gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.03,
                                max_depth=4, max_features='sqrt',
                                min_samples_leaf=15, min_samples_split=30,
                                loss='huber', random_state=5)
eNet = make_pipeline(RobustScaler(), ElasticNet(alpha=4.0, l1_ratio=0.005, random_state=3))
kr = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
xgbr = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

# score = cv(ridge)
# print("\nRidge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
#
# score = cv(lasso)
# print("Lasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
#
# score = cv(eNet)
# print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
#
# score = cv(kr)
# print("KernelRidge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
#
# score = cv(gbr)
# print("GradientBoostingRegressor score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
#
# score = cv(xgbr)
# print("XGBRegressor score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


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

    # Do the predictions of all base models on the test data and use the averaged predictions as
    # meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)


stacked_averaged_models = StackingAveragedModels(base_models=(eNet, gbr, ridge),
                                                 meta_model=lasso)

# score = cv(stacked_averaged_models)
# print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


stacked_averaged_models.fit(train.values, y_train)
stacked_train_pred = stacked_averaged_models.predict(train.values)
stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))
print(rmsle(y_train, stacked_train_pred))

xgbr.fit(train, y_train)
xgb_train_pred = xgbr.predict(train)
xgb_pred = np.expm1(xgbr.predict(test))
print(rmsle(y_train, xgb_train_pred))

gbr.fit(train, y_train)
gbr_train_pred = gbr.predict(train)
gbr_pred = np.expm1(gbr.predict(test))
print(rmsle(y_train, xgb_train_pred))

print('RMSLE score on train data:')
print(rmsle(y_train, stacked_train_pred*0.75 + xgb_train_pred*0.15 + gbr_train_pred*0.15))

ensemble = stacked_pred*0.75 + xgb_pred*0.15 + gbr_pred*0.15
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = ensemble
sub.to_csv('submission.csv', index=False)
