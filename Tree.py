import heapq
from copy import deepcopy
import math
import numpy as np
from Dataset import WordDataset
from tqdm import tqdm


class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._index = 0

    def push(self, item, priority):
        heapq.heappush(self._queue, (-priority, self._index, item))
        self._index += 1

    def pop(self):
        return heapq.heappop(self._queue)[-1]
    
    def is_empty(self) -> bool:
        return (len(self._queue) == 0)

def entropy(p: int):
    if (p == 0):
        return 0
    return -1 * p * math.log2(p)

def information_gain(y: np.ndarray[int]) -> float:
    class_1 = 0; class_2 = 0

    for val in y:
        if (val == 1):
            class_1 += 1
        else:
            class_2 += 1
    p0 = class_1 / len(y)
    p1 = class_2 / len(y)

    return entropy(p0) + entropy(p1)

def split_data(inputs: list[np.ndarray], targets: np.ndarray, given: int) -> tuple[list, list, list, list]:
    """
        given inputs and targets, split the data in which the inputs contain the given int

        retVal:
            inpust_have, targets_have, inputs_dont_have, targets_dont_have
    """
    inputs_1 = []; targets_1 = []
    inputs_2 = []; targets_2 = []
    
    for idx in range(len(inputs)):
        if (inputs[idx][given] == 1):
            n = deepcopy(inputs[idx])
            n[given] = -1
            inputs_1.append(n)
            targets_1.append(targets[idx])
        else:
            n = deepcopy(inputs[idx])
            n[given] = -1
            inputs_2.append(inputs[idx])
            targets_2.append(targets[idx])


    return inputs_1, targets_1, inputs_2, targets_2

def information_gain_by_split(inputs: list[np.ndarray], targets: list[int], given: int) -> float:
    if (inputs[0][given] == -1):
        return -1.0
    class_1_have = 0; class_2_have = 0
    class_1_no = 0; class_2_no = 0
    total_len = len(targets)
    for idx in range(len(inputs)):

        if (inputs[idx][given] == 1):
            if (targets[idx] == 1):
                class_1_have += 1
            else:
                class_2_have += 1
        else:
            if (targets[idx] == 1):
                class_1_no += 1
            else:
                class_2_no += 1

    E = information_gain(targets)
    have_denom = 0
    if (class_1_have + class_2_have != 0):
        have_denom = (1 / (class_1_have + class_2_have))
    p0_have = class_1_have * have_denom
    p1_have = class_2_have * have_denom

    no_denom = 0
    if (class_1_no + class_2_no != 0):
        no_denom = 1 / (class_1_no + class_2_no)
    p0_no = class_1_no * no_denom
    p1_no = class_2_no * no_denom

    E_split = ((class_1_have + class_2_have) / total_len) * (entropy(p0_have) + entropy(p1_have)) \
             + ((class_1_no + class_2_no) / total_len) * (entropy(p0_no) + entropy(p1_no))

    return E - E_split 

class TreeNode:
    children: list # list of TreeEdges 
    feature = 0         # feature to do the split on
    ig_val = 0          # information gain value
    inputs : list[np.ndarray[int]]
    targets : list[int]
    prediction = -1
    depth: int

    def __init__(self, inputs, targets, depth) -> None:
        self.inputs = inputs
        self.targets = targets
        self.children = []
        self.depth = depth
    
    def add_child(self, child):
        self.children.append(child)

    def find_optimal_split(self, feature_range) -> tuple[int, float]:
        """
            finds the optimal feature to do the split, and then stores that in feature and the score
            on ig_val fields of itself

            retVal:
                feature, split_score
        """

        words = range(feature_range)
        
        self.feature = words[0]
        self.ig_val = information_gain_by_split(self.inputs, self.targets, words[0])
        for word in words:
            val = information_gain_by_split(self.inputs, self.targets, word)
            if (val > self.ig_val):
                self.feature = word
                self.ig_val = val
        
        return self.feature, self.ig_val
    
    def predict(self, given_input) -> int:
        """
            given an input will return the predicted value
        """
        # if internal node
        if (len(self.children) != 0):
            for child in self.children:
                if(given_input[self.feature]== child.val):
                    return child.dest.predict(given_input)
                
        #if leaf node        
        if (self.prediction == -1):
            p = self.targets.count(1) / len(self.targets)
            if (p >= 0.5):
                self.prediction = 1
            else:
                self.prediction = 2
        return self.prediction
    
    def get_depth(self) -> int:
        depth = 1
        for child in self.children:
            depth = max(depth, 1 + child.dest.get_depth())

        return depth
    
    def printout(self, max_depth, wds: WordDataset):
        return


  
class TreeEdge:
    dest: TreeNode = None
    val = 0

    def __init__(self, dest, val) -> None:
        self.dest = dest
        self.val = val

class DesicionTree:
    root: TreeNode = None
    heap = PriorityQueue()
    max_size: int
    size = 0

    def __init__(self, max_size=100):
        self.max_size=max_size

    def fit(self, inputs: list[np.ndarray[int]], targets: list[int], words: WordDataset):
        if (self.root != None):
            raise RuntimeError(" supposed to use fit method only once after object init")
        
        self.root = TreeNode(deepcopy(inputs), deepcopy(targets), 1)
        feature, split_val = self.root.find_optimal_split(len(words.words))
        self.heap.push(self.root, split_val)

        progress_bar = tqdm(total=self.max_size, desc="Processed Nodes")
        while(self.heap.is_empty() == False and self.size <= self.max_size):
            node: TreeNode = self.heap.pop()
            # create the new nodes
            x_left, y_left, x_right, y_right = split_data(node.inputs, node.targets, node.feature)
            left_node = TreeNode(x_left, y_left, node.depth + 1)
            right_node = TreeNode(x_right, y_right, node.depth + 1)
            # push the new nodes into the heap
            l_f, l_val = left_node.find_optimal_split(len(words.words))
            r_f, r_val = right_node.find_optimal_split(len(words.words))
            self.heap.push(left_node, l_val)
            self.heap.push(right_node, r_val)
            # add new nodes as children
            left_branch = TreeEdge(left_node, 0)
            right_branch = TreeEdge(right_node, 1)

            node.add_child(left_branch)
            node.add_child(right_branch)
            progress_bar.update(1)

            # increase size
            self.size += 1
        
        progress_bar.close()
            
    def predict(self, inputs: list[np.ndarray[int]]) -> list[int]:
        return self(inputs)

    def __call__(self, inputs: list[np.ndarray[int]]) -> list[int]:
        return [self.root.predict(x) for x in inputs]
    
    def get_printout(self, wds: WordDataset):
        self.root.printout(self.root.get_depth())

