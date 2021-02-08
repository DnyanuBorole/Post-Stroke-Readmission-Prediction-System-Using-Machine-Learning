from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import mysql.connector
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
# from sklearn.metrics import plot_confusion_matrix
# import matplotlib.pyplot as plt
def classify_rfc(x_train, x_test, y_train, y_test):
        mydb = mysql.connector.connect(host="localhost", user="root", password="root", database='stroke')
        cursor = mydb.cursor()
        classify2 = RandomForestClassifier()
        classify2.fit(x_train, y_train)
        predicted2 = classify2.predict(x_test)
        query = "INSERT INTO  performance  VALUES (%s,%s,%s,%s,%s,%s)"

        RFC = metrics.accuracy_score(y_test, predicted2) * 100
        print("The accuracy score using the RFC is ->")
        print(metrics.accuracy_score(y_test, predicted2))

        precision = metrics.precision_score(y_test, predicted2, average='macro', pos_label='1')
        print("Precision score of RFC IS ->")
        print(metrics.precision_score(y_test, predicted2, average='micro', pos_label='1'))

        recall = metrics.recall_score(y_test, predicted2, average='macro', pos_label='1')
        print("Recall score of RFC  IS ->")
        print(metrics.recall_score(y_test, predicted2, average='micro', pos_label='1'))

        f1_score = metrics.f1_score(y_test, predicted2, average='macro', pos_label='1')
        print("F1 score of RFC IS ->")
        print(metrics.f1_score(y_test, predicted2, average='micro', pos_label='1'))

        roc_score = metrics.roc_auc_score(y_test, predicted2, average='macro')
        print("roc_score of RFC IS ->")
        print(metrics.roc_auc_score(y_test, predicted2, average='micro'))
        print('---------------------------------------------- ')
        # plot_confusion_matrix(classify2, x_test, y_test)
        # plt.title('Confusion Matrix of RFC')
        # plt.show()
        val = ("RFC", str(RFC), str(precision), str(recall), str(f1_score), str(roc_score))
        cursor.execute(query, val)
        mydb.commit()
        cursor.close()
        mydb.close()
        return RFC