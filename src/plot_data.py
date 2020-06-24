import seaborn as sb
import matplotlib.pyplot as plt

def plot_confusion_matrix(data, label):
    sb.heatmap(data, annot = True, xticklabels = label, yticklabels = label, cmap="YlGnBu", fmt='g')
    plt.show()