
# IoMT_ML_Benchmark.py
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

LABEL_COLUMN='Label'

def load_data(path):
    df=pd.read_csv(path)
    X=df.drop(columns=[LABEL_COLUMN]).select_dtypes(include=['number'])
    y=LabelEncoder().fit_transform(df[LABEL_COLUMN])
    return X,y

def evaluate(name,model,Xtr,Xte,ytr,yte):
    model.fit(Xtr,ytr)
    pred=model.predict(Xte)
    acc=accuracy_score(yte,pred)
    p,r,f1,_=precision_recall_fscore_support(yte,pred,average='weighted',zero_division=0)
    print('\n',name)
    print('Accuracy:',acc)
    print(classification_report(yte,pred))
    return [name,acc,p,r,f1]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',required=True)
    args=ap.parse_args()

    X,y=load_data(args.data)

    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)

    imp=SimpleImputer(strategy='median')
    sc=StandardScaler()

    Xtr=sc.fit_transform(imp.fit_transform(Xtr))
    Xte=sc.transform(imp.transform(Xte))

    models={
      'Logistic Regression':LogisticRegression(max_iter=1000),
      'Decision Tree':DecisionTreeClassifier(random_state=42),
      'Random Forest':RandomForestClassifier(n_estimators=200,random_state=42),
      'SVM':SVC(),
      'KNN':KNeighborsClassifier(),
      'Gaussian NB':GaussianNB(),
      'AdaBoost':AdaBoostClassifier(random_state=42)
    }

    results=[]
    for n,m in models.items():
        results.append(evaluate(n,m,Xtr,Xte,ytr,yte))

    pd.DataFrame(results,columns=['Model','Accuracy','Precision','Recall','F1']).to_csv('ML_Model_Comparison.csv',index=False)
    print('Saved ML_Model_Comparison.csv')

if __name__=='__main__':
    main()
