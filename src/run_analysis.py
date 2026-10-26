from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'/'student_performance_dataset.csv'; OUT=ROOT/'outputs'
(OUT/'visualizations').mkdir(parents=True,exist_ok=True); (OUT/'evaluation_results').mkdir(parents=True,exist_ok=True)
df=pd.read_csv(DATA).drop_duplicates().copy(); features=['study_hours_per_day','attendance_pct','previous_exam_marks','assignment_score']; target='final_exam_marks'
for c in features+[target]: df[c]=pd.to_numeric(df[c],errors='coerce')
df=df.dropna(subset=[target]); df[features+[target]].describe().T.to_csv(OUT/'evaluation_results'/'descriptive_statistics.csv')
sns.set_theme()
for x,name in [('study_hours_per_day','study_hours_vs_marks'),('attendance_pct','attendance_scatter')]:
    plt.figure(figsize=(7,4)); sns.scatterplot(data=df,x=x,y=target); plt.tight_layout(); plt.savefig(OUT/'visualizations'/(name+'.png'),dpi=140); plt.close()
plt.figure(figsize=(7,4)); sns.histplot(df[target],kde=True); plt.tight_layout(); plt.savefig(OUT/'visualizations'/'final_marks_histogram.png',dpi=140); plt.close()
plt.figure(figsize=(7,4)); sns.boxplot(data=df,y=target); plt.tight_layout(); plt.savefig(OUT/'visualizations'/'final_marks_boxplot.png',dpi=140); plt.close()
plt.figure(figsize=(7,5)); sns.heatmap(df[features+[target]].corr(),annot=True,cmap='coolwarm'); plt.tight_layout(); plt.savefig(OUT/'visualizations'/'correlation_heatmap.png',dpi=140); plt.close()
X,y=df[features],df[target]; Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
pipe=Pipeline([('imputer',SimpleImputer(strategy='median')),('model',LinearRegression())]); pipe.fit(Xtr,ytr); pred=pipe.predict(Xte)
metrics={'MAE':float(mean_absolute_error(yte,pred)),'RMSE':float(np.sqrt(mean_squared_error(yte,pred))),'R2':float(r2_score(yte,pred)),'train_rows':int(len(Xtr)),'test_rows':int(len(Xte))}
(OUT/'evaluation_results'/'regression_metrics.json').write_text(json.dumps(metrics,indent=2)); pd.DataFrame({'actual_marks':yte.to_numpy(),'predicted_marks':np.round(pred,2)}).to_csv(OUT/'evaluation_results'/'actual_vs_predicted.csv',index=False)
cv=cross_val_score(pipe,X,y,cv=5,scoring='neg_mean_absolute_error'); (OUT/'evaluation_results'/'cross_validation.json').write_text(json.dumps({'5_fold_MAE_mean':float(-cv.mean()),'5_fold_MAE_std':float(cv.std())},indent=2))
ybin=(df[target]>=40).astype(int); a,b,c,d=train_test_split(X,ybin,test_size=.2,random_state=42,stratify=ybin); clf=Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler()),('model',LogisticRegression(max_iter=1000))]); clf.fit(a,c); (OUT/'evaluation_results'/'optional_pass_fail_metrics.json').write_text(json.dumps({'accuracy':float(accuracy_score(d,clf.predict(b)))},indent=2))
print('Regression metrics:',metrics)