"""
survival 	Survival 	0 = No, 1 = Yes
pclass 	Ticket class 	1 = 1st, 2 = 2nd, 3 = 3rd
sex 	Sex male female
name
Age 	Age in years
sibsp 	# of siblings / spouses aboard the Titanic
parch 	# of parents / children aboard the Titanic
ticket 	Ticket number  'A/5 21171'   'STON/O2. 3101282'   '113803' 'PC 12'
fare 	Passenger fare
cabin 	Cabin number  'C85'  'C123'  'C23 C25 C27'  'A6' 'D56'
embarked 	Port of Embarkation 	C = Cherbourg, Q = Queenstown, S = Southampton
"""
from pandas import read_csv, get_dummies, concat, DataFrame
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier


def make_one_hot(data, field):
    temp = get_dummies(data[field], prefix=field)
    data.drop(field, axis=1, inplace=True)
    return concat([data, temp], axis=1)


def adjust_title(title):
    if title in ['Mr', 'Miss', 'Mrs', 'Ms', 'Sir', 'Lady', 'the Countess']:
        return 'English'
    elif title in ['Major', 'Col', 'Capt']:
        return 'Military'
    elif title in ['Mlle', 'Mme', 'Don', 'Dona', 'Jonkheer']:
        return 'OtherEuropean'
    else:
        return title


def feature_engineering(data):
    data.Cabin.fillna('U', inplace=True) # U is fake
    data.Cabin = data.Cabin.map(lambda c: c[0] if c[0] != 'T' else 'U')  # one problematic T in train

    data['FamilySize'] = data.Parch + data.SibSp + 1
    data['Age'].fillna(-1, inplace=True)
    data.Name = data.Name.map(lambda name: name.split(',')[1].split('.')[0].strip()).map(adjust_title)
    data.Age.fillna(data.Age.mean(), inplace=True)
    data['Age'] = data.Age.map(int)
    data.Fare.fillna(data['Fare'].mean(), inplace=True)
    data['Fare'] = data['Fare'] / data['FamilySize']
    data['Ticket'] = data['Ticket'].map(lambda c: c.split()[0].replace('.', ''))
    data['Ticket'] = data['Ticket'].map(lambda c: 'X' if c.isdigit() else c)
    data.Embarked.fillna('C', inplace=True)
    for col in ['Cabin', 'Ticket', 'Sex', 'Embarked', 'Name']:
        data = make_one_hot(data, col)
    data.drop(['Cabin_U', 'SibSp', 'Parch', 'Ticket_X', 'Age'], axis=1, inplace=True)
    return data


x_train = read_csv('titanic/train.csv', sep=',')
x_test = read_csv('titanic/test.csv', sep=',')

y_train = x_train.Survived
PassengerId = x_test.PassengerId.to_numpy()

x_train.drop(['PassengerId', 'Survived'], axis=1, inplace=True)
x_test.drop(['PassengerId'], axis=1, inplace=True)
all_data = concat([x_train, x_test])
all_data = feature_engineering(all_data)
x_train = all_data[:len(x_train)]
x_test = all_data[len(x_train):]
clf = MLPClassifier()
clf.fit(x_train, y_train)
scores = cross_val_score(clf, x_train, y_train, cv=5)
print("Accuracy: {:0.2f} (+/- {:0.2f})".format(scores.mean(), scores.std() * 2))
predictions = clf.predict(x_test)
df = DataFrame(data=list(zip(PassengerId, predictions)), columns=['PassengerId', 'Survived'])
df.to_csv('res.csv', index=False)
