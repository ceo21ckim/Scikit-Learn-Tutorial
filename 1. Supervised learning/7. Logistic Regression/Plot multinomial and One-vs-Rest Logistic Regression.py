import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import make_blobs 
from sklearn.linear_model import LogisticRegression

centers = np.array([[-5, 0], [0, 1.5], [5, -1]])

x, y = make_blobs(n_samples = 1000, centers = centers, random_state = 40)
transformation = np.array([[0.4, 0.2], [-0.4, 1.2]])

x = np.dot(x, transformation)


for multi_class in ('multinomial', 'ovr'):
    clf = LogisticRegression(
        solver = 'sag', max_iter = 100, random_state = 42, multi_class = multi_class # multinomial, ovr
    ).fit(x, y)
    
    print(f"training score : {clf.score(x, y):.3f} {(multi_class)}")
    
    h = 0.02 
    x_min, x_max = x[:,0].min() - 1, x[:,0].max() + 1
    y_min, y_max = x[:,1].min() - 1, x[:,1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # np.c_ concatnate(axis = 1)
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    
    plt.figure()
    plt.contourf(xx, yy, z, cmap = plt.cm.Paired)
    plt.title("Decision surface of LogisticRegression (%s)" % multi_class)
    plt.axis('tight')
    
    colors = 'bry'
    for i, color in zip(clf.classes_, colors):
        idx = np.where(y == i)
        plt.scatter(
            x[idx, 0], x[idx, 1], c = color, cmap = plt.cm.Paired, edgecolor = 'black', s = 20
        )
        
        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()
        coef = clf.coef_ 
        intercept = clf.intercept_
        
        def plot_hyperplane(c, color):
            def line(x0):
                return (-(x0 * coef[c, 0]) - intercept[c]) / coef[c, 1]
            plt.plot([xmin, xmax], [line(xmin), line(xmax)], ls = '--', color = color)
            
        for i, color in zip(clf.classes_, colors):
            plot_hyperplane(i, color)
            
plt.show()

