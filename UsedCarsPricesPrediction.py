import numpy as np

import matplotlib.pyplot as plt

x = np.array([[1980],[1983],[1984],[1990],[1994]])

y = np.array([[1500],[1580],[1850],[3520],[4000]])

plt.scatter(x ,y)

plt.title("Used Cars Prices", fontsize=24)

plt.xlabel("X axes", fontsize=14)

plt.ylabel("Y axes", fontsize=14)

plt.grid(True)

plt.show()

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x,y)

array1 = np.array([1999])
array2 = array1.reshape(1, -1)

print(model.predict(array2))

x_test = np.array([[1975],[1985],[1988],[2001],[1993]])

y_test = np.array([[1200],[1630],[1750],[5520],[3500]])

model.score(x_test,y_test)
