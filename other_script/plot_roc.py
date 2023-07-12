
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import auc,roc_curve
import matplotlib.pyplot as plt

def plot_roc_cv(y_test, predict_x_test, name, out):
    RocCurveDisplay.from_predictions(y_test, predict_x_test, name=name,
                                     color = "darkorange",)
    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    # plt.title("One-vs-Rest ROC curves:\nVirginica vs (Setosa & Versicolor)")
    plt.legend()
    plt.show()
    plt.savefig(out)