#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
import joblib
from car_data_prep import prepare_data

# טעינת הנתונים
df = pd.read_csv('https://raw.githubusercontent.com/raza783/cars-project-part-2/main/dataset.csv')

# עיבוד הנתונים
df_processed = prepare_data(df)

# הגדרת העמודות הרלוונטיות
features = ['manufactor', 'Year', 'model', 'Hand', 'Gear', 'capacity_Engine', 'Engine_type', 
            'Prev_ownership', 'Curr_ownership', 'Area', 'City', 'Km', 'Test', 
            'Supply_score', 'Pic_num', 'Color']
target = 'Price'

X = df_processed[features]
y = df_processed[target]

# הגדרת עמודות מספריות וקטגוריות
numeric_features = ['Year', 'Hand', 'capacity_Engine', 'Km', 'Test', 'Supply_score', 'Pic_num']
categorical_features = ['manufactor', 'model', 'Gear', 'Engine_type', 'Area', 'City', 'Color', 'Prev_ownership', 'Curr_ownership']

# הגדרת טרנספורמרים
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)],
    remainder='passthrough')

# עיבוד נתוני ה-X
X = preprocessor.fit_transform(X)

# המרת הנתונים למערך דחוס כדי לטפל בערכי NaN וערכים אינסופיים
X = X.toarray()

# המרת הנתונים ל-DataFrame כדי לטפל בערכי NaN וערכים אינסופיים
X = pd.DataFrame(X).astype(float)

# מילוי ערכי NaN במדיאן של העמודה
X.fillna(X.median(), inplace=True)

# החלפת ערכים אינסופיים בערך המקסימלי בעמודה
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.median(), inplace=True)

# הגדרת המודל
model = ElasticNet(random_state=42)

# הגדרת רשת הפרמטרים לחיפוש
param_grid = {
    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
    'l1_ratio': [0.1, 0.5, 0.7, 0.9, 1.0]
}

# הגדרת GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=10, n_jobs=-1)

# ביצוע החיפוש
grid_search.fit(X, y)

# הצגת הפרמטרים הטובים ביותר שנמצאו
print("Best parameters found: ", grid_search.best_params_)
print("Best RMSE: ", -grid_search.best_score_)

# אימון המודל עם הפרמטרים הטובים ביותר
best_model = grid_search.best_estimator_
best_model.fit(X, y)

# שמירת המודל המאומן לקובץ PKL
joblib.dump(best_model, 'trained_model.pkl')

