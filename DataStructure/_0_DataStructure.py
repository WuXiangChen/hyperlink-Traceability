
from typing import List
import numpy as np

class ArtifactHyperLink:
    def __init__(self, artifactIdList, hyperedge_set, artifact_index):
        self.artifactIdList = artifactIdList
        self.hyperedge_set = hyperedge_set
        self.artifact_index = artifact_index

    def create_incidence_matrix(self):
        # Initialize the matrix
        num_hyperedges = len(self.hyperedge_set)
        num_artifacts = len(self.artifactIdList)
        matrix = np.zeros((num_artifacts, num_hyperedges), dtype=int)
        for hyperedge_idx, hyperedge in enumerate(self.hyperedge_set):
            for artifactId in hyperedge:
                if artifactId in self.artifact_index:
                    matrix[self.artifact_index[artifactId], hyperedge_idx] = 1
                else:
                    raise Exception(f"Artifact {artifactId} is not in the artifactDict!")
        return matrix