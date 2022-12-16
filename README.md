# opensourcesw-final

## Configuration instructions
### I used Voting Classifier with...
1. KNN
```python
from sklearn.neighbors import KNeighborsClassifier
knn_clf1 = KNeighborsClassifier(n_neighbors=1)
```
2. SVM
```python
from sklearn.svm import SVC
svc_clf1 = SVC(kernel='rbf', C=10, coef0=5,degree=9,gamma=1.2,probability=True)
svc_clf2 = SVC(kernel='rbf', C=10, coef0=8,degree=3,gamma=1.4,probability=True)
svc_clf3 = SVC(kernel='rbf', C=10, coef0=12,degree=6,gamma=1.0,probability=True)
svc_clf4 = SVC(kernel='rbf', C=10, coef0=10,degree=11,gamma=1.8,probability=True)
```
3. Extra Trees
```python
from sklearn.ensemble import ExtraTreesClassifier
xtree_clf1 = ExtraTreesClassifier(n_estimators=3, random_state=1898,criterion='entropy')
xtree_clf2 = ExtraTreesClassifier(n_estimators=3, random_state=7567,criterion='entropy')
xtree_clf3 = ExtraTreesClassifier(n_estimators=3, random_state=9244,criterion='entropy')
xtree_clf4 = ExtraTreesClassifier(n_estimators=3, random_state=5147,criterion='entropy')
xtree_clf5 = ExtraTreesClassifier(n_estimators=3, random_state=6161,criterion='entropy')
```
4. Voting
```python
from sklearn.ensemble import VotingClassifier
vote = VotingClassifier(estimators=[
                                    ('KNN1',knn_clf1),
                                    ('svc1',svc_clf1),
                                    ('svc2',svc_clf2),
                                    ('svc3',svc_clf3),
                                    ('svc4',svc_clf4),
                                    ('XTree1',xtree_clf1),
                                    ('XTree2',xtree_clf2),
                                    ('XTree3',xtree_clf3),
                                    ('XTree4',xtree_clf4),
                                    ('XTree5',xtree_clf5),
                                   ])
```

## Operating instructions
- Please run from the top cell in order.
- I uploaded .pkl file. Please run the file.
```python
import pickle
with open('./round_3.pkl',"rb") as fr:
    model = pickle.load(fr)
y_pred = model.predict(X_test)
```

## Copyright and licensing information
- MIT License

## Contact information
- Name: 김태환
- Student ID: 20225126
- E-mail: dmkthwan11@gmail.com
