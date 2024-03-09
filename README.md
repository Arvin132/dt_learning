# DTlearning 🌳

This is a basic Desicion Tree learning class algorithm that uses information gain to split nodes.

there is an example dataset in the repository that can be used to train the tree to classify a reddit post to two classes (r/athesim and r/books)

## example use

```python
from Tree import DesicionTree

dt = DesicionTree(amount_of_nodes)
dt.fit(train_inputs, train_targets, words_Dataset)

predictions = dt.predict(test_inputs)
```



