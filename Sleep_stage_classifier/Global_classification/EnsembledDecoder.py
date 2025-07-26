from typing import List, Union, Tuple
import copy
import numpy as np

class TreeNode:
    def __init__(self, classifier_index: int = None, positive=None, negative=None, class_label=None, class_label_friendly_name=None):
        self.classifier_index = classifier_index  # Index in the input encoding list
        self.positive = positive                  # TreeNode or class label
        self.negative = negative                  # TreeNode or class label
        self.class_label = class_label            # Final classification result if it's a leaf
        self.class_label_friendly_name = class_label_friendly_name

        self.true_n_classes = None

    def is_leaf(self):
        return self.class_label is not None
    
    def __call__(self, x: np.ndarray):
        """
        x: 1D numpy contains only 0 or 1 for each classifier, shape: (n_clf,)
        """
        if self.is_leaf():
            return self.class_label
        
        if x[self.classifier_index]:
            return self.positive(x)
        else:
            return self.negative(x)
        
    def __repr__(self):
        if self.true_n_classes is not None:
            return f'True n_classes: {self.true_n_classes}\n' + \
                f'Node(clf_{self.classifier_index}, +->{self.positive}, -->{self.negative})'
        
        if self.is_leaf():
            # return f'Leaf({self.class_label_friendly_name})'
            return f'Leaf({self.class_label})'
        

        return f'Node(clf_{self.classifier_index}, +->{self.positive}, -->{self.negative})'


def build_decoder_tree(encoding_list: List[str], class_labels: List[str]) -> TreeNode:
    n_classes = len(set(class_labels))
    true_n_classes = n_classes

    # Each node will try to split remaining possible classes
    def recurse(possible_classes: List[int], available_clfs: List[Tuple[int, str]]) -> TreeNode:
        nonlocal true_n_classes  # Declare as nonlocal to modify the outer variable
        
        if len(possible_classes) == 1:
            return TreeNode(
                class_label=possible_classes[0],
                class_label_friendly_name=class_labels[possible_classes[0]])

        for clf_index, enc in available_clfs:
            pos_group = []
            neg_group = []
            unknown_group = []

            for c in possible_classes:
                v = enc[c]
                if v == '1':
                    pos_group.append(c)
                elif v == '0':
                    neg_group.append(c)
                else:  # 'n'
                    unknown_group.append(c)

            if unknown_group:
                continue  # Can't split unless all current classes are either 1 or 0

            if not pos_group or not neg_group:
                continue  # Not a valid split

            # Recurse on subtrees with remaining classifiers
            remaining_clfs = [(i, e) for i, e in available_clfs if i != clf_index]
            return TreeNode(
                classifier_index=clf_index,
                positive=recurse(pos_group, remaining_clfs),
                negative=recurse(neg_group, remaining_clfs)
            )

        true_n_classes -= 1

        # raise ValueError(f"Cannot uniquely determine all classes: {[class_labels[i] for i in possible_classes]}")
        print(f"Warning: No valid split found for classes {possible_classes} with encodings {available_clfs}")
        return TreeNode(
            class_label=true_n_classes,
            class_label_friendly_name=",".join([str(class_labels[c]) for c in possible_classes]),
        )

    all_classes = list(range(n_classes))
    indexed_encodings = list(enumerate(encoding_list))
    tree = recurse(all_classes, indexed_encodings)
    tree.true_n_classes = true_n_classes
    return tree

if __name__ == "__main__":
    # Assume 5 classes: A=0, B=1, C=2, D=3, E=4
    encoding_list = [
        '01000',  # clf_0: B vs others
        '11100',  # clf_1: A/B/C vs D/E
        '1n000',  # clf_2: A vs C
        '00010',  # clf_3: D vs E
    ]

    class_names = ['A', 'B', 'C', 'D', 'E']

    tree = build_decoder_tree(encoding_list, class_names)
    print(tree)
