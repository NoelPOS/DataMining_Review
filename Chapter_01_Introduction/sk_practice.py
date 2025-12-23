from sklearn.datasets import load_iris
dataset = load_iris()
X = dataset.data
y = dataset.target

print(dataset.DESCR)
print(X)
print(y)