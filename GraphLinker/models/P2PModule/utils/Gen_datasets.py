from turtle import pos
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

def generate_negative_samples(num_a, pos_data, type_ ):
    """
    Generate negative samples dynamically.
    """
    neg_candidates = []
    # 更改这里的负样本生成方式, 随机替换pos_data的第一个元素作为负样本
    for pair in pos_data:
        while True:
            new_pair = [random.choice(range(num_a)), pair[1], type_]
            if new_pair in pos_data and new_pair not in neg_candidates:
                neg_candidates.append(new_pair)
                break
    return np.array(neg_candidates)

def generate_train_val_test_splits(issue_pr_data, issue_issue_data, pr_pr_data, issue_num, pr_num, k=5, neg_ratio=[0.4, 0.3, 0.3]):
    """
    Dynamically generate train, val, and test splits for k-fold cross-validation.
    Returns a generator that yields splits for each fold.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    all_pos_data = np.vstack([issue_pr_data, issue_issue_data, pr_pr_data])
    # Generate negative samples dynamically
    neg_candidates_pr_issue = generate_negative_samples(issue_num, issue_pr_data, type_=0)
    neg_candidates_issue_issue = generate_negative_samples(issue_num, issue_issue_data, type_=1)
    neg_candidates_pr_pr = generate_negative_samples(pr_num,  pr_pr_data, type_=2)

    # Add labels and combine positive and negative samples
    pos_labeled = np.hstack((all_pos_data, np.ones((len(all_pos_data), 1))))  # Label positive samples as 1
    neg_pr_issue_labeled = np.hstack((neg_candidates_pr_issue, np.zeros((len(neg_candidates_pr_issue), 1))))  # Label as 0
    neg_issue_issue_labeled = np.hstack((neg_candidates_issue_issue, np.zeros((len(neg_candidates_issue_issue), 1))))  # Label as 0
    neg_pr_pr_labeled = np.hstack((neg_candidates_pr_pr, np.zeros((len(neg_candidates_pr_pr), 1))))  # Label as 0

    all_data = np.vstack([pos_labeled, neg_pr_issue_labeled, neg_issue_issue_labeled, neg_pr_pr_labeled])
    all_data = shuffle(all_data, random_state=42)
    
    # Perform KFold splitting
    for train_index, test_index in kf.split(all_data):
        train_data = all_data[train_index]
        test_data = all_data[test_index]

        # Split into features and labels
        train_features, train_labels = train_data[:, :-1], train_data[:, -1]
        test_features, test_labels = test_data[:, :-1], test_data[:, -1]

        # Separate positive and negative samples for train and test sets
        train_pos = train_features[train_labels == 1].astype(int)
        train_neg = train_features[train_labels == 0].astype(int)
        test_pos = test_features[test_labels == 1].astype(int)
        test_neg = test_features[test_labels == 0].astype(int)

        # Return in the desired format
        yield {
            'train_pos': train_pos,
            'train_neg': train_neg,
            'test_pos': test_pos,
            'test_neg': test_neg
        }

def prepare_datasets(repoName, k=5):
    """
    Prepare a unified dataset dynamically without saving intermediate results.
    Supports k-fold cross-validation for all sources of data.
    """
    rootPath = f"dataset/{repoName}/"

    # Define file paths
    issue_pr_path = rootPath + '/Index/issue_pr_index.txt'
    issue_issue_path = rootPath + '/Index/issue_issue_index.txt'
    pr_pr_path = rootPath + '/Index/pr_pr_index.txt'

    # Read issue and PR counts
    issue_num = len(pd.read_csv(rootPath + '/Index/issue_index.txt', header=None))
    pr_num = len(pd.read_csv(rootPath + '/Index/pr_index.txt', header=None))

    # Read all datasets and add source identifiers
    def read_data_with_source(file_path, source_label):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        # Add a source label to each pair
        data = [list(map(int, line.strip().split(" "))) + [source_label] for line in lines if line.strip() != '']
        return np.array(data)

    # Load and label datasets
    issue_pr_data = read_data_with_source(issue_pr_path, source_label=0)
    issue_issue_data = read_data_with_source(issue_issue_path, source_label=1)
    pr_pr_data = read_data_with_source(pr_pr_path, source_label=2)

    # Generate train, val, and test splits dynamically
    datasets = list(generate_train_val_test_splits(issue_pr_data, issue_issue_data, pr_pr_data, issue_num, pr_num, k=k))
    return datasets


# Example usage
'''
if __name__ == "__main__":
    rootPath = "dataset/symfony/"
    repoName = "example_repo"

    # Prepare datasets for 5-fold cross-validation
    datasets = prepare_datasets(rootPath, repoName, k=5)

    for fold, dataset in enumerate(datasets):
        print(f"Fold {fold + 1}:")
        print("Train Pos:", dataset['train_pos'].shape)
        print("Train Neg:", dataset['train_neg'].shape)
        print("Test Pos:", dataset['test_pos'].shape)
        print("Test Neg:", dataset['test_neg'].shape)
'''
