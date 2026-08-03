#Importando dataset 
import pandas as pd

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)


###########################################################
#Melhora na distribuição dos valores
import numpy as np
df["Amount_log"] = np.log1p(df["Amount"]) 


##########################################################
#Dividindo dados para treino e para teste
from sklearn.model_selection import train_test_split 

x = df.drop("Class", axis=1)
y = df["Class"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, stratify=y, test_size=0.3, random_state=48
)


#########################################################
#Pipeline de machine learning com balanceamento de classes
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import ADASYN
from xgboost import XGBClassifier

pipeline = Pipeline([
    ("adasyn", ADASYN(
        sampling_strategy=0.6,
        random_state=48
    )),
    ("model", XGBClassifier(
        scale_pos_weight=10,
        random_state=48
    ))
])

pipeline.fit(x_train, y_train)

y_prob = pipeline.predict_proba(x_test)[:, 1]

threshold = 0.8

y_pred = (y_prob >= threshold).astype(int)


##########################################################
#Análises para avaliação do modelo:

#Recall médio de seeds variadas com validação cruzada:
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(
    n_splits=20,
    shuffle=True,
    random_state=48
)

scores = cross_val_score(
    pipeline,
    x,
    y,
    cv=cv,
    scoring="recall",
    n_jobs= -1
)

print("Recall de cada fold: ", scores)
print("Recall médio: ", scores.mean())
print("Desvio padrão do recall: ", scores.std())

#Classification report:
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


#Matriz de confusão:
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred) 

print(cm)