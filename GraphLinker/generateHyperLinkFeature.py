from ast import Set
from itertools import combinations
import os
import torch
from typing import Counter, Dict, List
import numpy as np
import pandas as pd
import numpy as np
from typing import Dict, List
from utils import Utils
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data, Batch

class generateHyperLinkFeature:
    def __init__(self):
        pass

    def generateHyperLinkFeature(self, hyperlinks: List[List[Dict]],  arts: Dict[int, Dict]):
        torch_fcgs = []
        for k,hyperlink in enumerate(hyperlinks):
            art_index = hyperlink[-1].item()
            art = arts[art_index]
            authorSet = art.globalDict["allAuthors"]
            allLabels = art.globalDict["allLabels"]
            allFileNames = art.globalDict["allFileNames"]
            allArtifactTypes = art.globalDict["allArtifactTypes"]

            hyperlink = hyperlink[:-1]
            edge = []
            for node_id in hyperlink:
                if node_id == -1:
                    continue
                edge.append(node_id.item())
            if len(edge)>50:
                edge = edge[:50]
                
            hCG = Utils.get_fully_network([edge])[0]
            ab = self.getAuthorVector(authorSet, edge=edge, artfact=art)
            lb = self.getLabelVector(allLabels, edge=edge, artfact=art)
            fb = self.getFileNameVector(allFileNames, edge=edge, artfact=art)
            atb = self.getArtifactTypeVector(allArtifactTypes, edge=edge, artfact=art)
            for i, node in enumerate(hCG.nodes):
                hCG.nodes[node]['ab'] = ab[i,:]  
                hCG.nodes[node]['lb'] = lb[i,:] 
                hCG.nodes[node]['fb'] = fb[i,:]
                hCG.nodes[node]['atb'] = atb[i,:]
                hCG.nodes[node]['batch'] = k
            hCG = from_networkx(hCG, group_node_attrs=["ab", "lb", 'fb', 'atb', 'batch'])
            torch_fcgs.append(hCG)
        data = Batch.from_data_list(torch_fcgs)
        return data

    def getAuthorVector(self, authorSet: Set, edge:List, artfact):
        # 根据edge中的所有节点建立AuthorVector
        hlFea = []
        for node in edge:
            author = artfact.getArtifactFeature(artifact_id=node, feature="author")
            reviewer = artfact.getArtifactFeature(artifact_id=node, feature="reviewer")
            closed_by = artfact.getArtifactFeature(artifact_id=node, feature="closed_by")
            # if author!=None and reviewer!=None:
            #     authors = [author] + reviewer
            # else:
            #     hlFea.append([0]*len(authorSet))
            #     continue
            authors = [author]
            if closed_by!=None:
                authors += [closed_by]
            author_count = Counter(authors)
            authorVector = [author_count.get(author, 0) for author in authorSet]
            hlFea.append(authorVector)
        hlFea_tensor = torch.tensor(hlFea)
        return hlFea_tensor
    
    def getLabelVector(self, LabelSet: Set, edge:List, artfact):
        # 根据edge中的所有节点建立LabelVector
        hlFea = []
        for node in edge:
            labels = artfact.getArtifactFeature(artifact_id=node, feature="labels")
            if labels==[] or labels==None:
                hlFea.append([0]*len(LabelSet))
                continue
            label_count = Counter(labels)
            labelVector = [label_count.get(label, 0) for label in LabelSet]
            hlFea.append(labelVector)
        hlFea_tensor = torch.tensor(hlFea)
        return hlFea_tensor
    
    def getFileNameVector(self, fileNameSet: Set, edge:List, artfact):
        # 根据edge中的所有节点建立FileNameVector
        hlFea = []
        for node in edge:
            fn = artfact.getArtifactFeature(artifact_id=node, feature="files")
            if fn==[] or fn==None:
                hlFea.append([0]*len(fileNameSet))
                continue
            fn_count = Counter(fn)
            fnVector = [fn_count.get(fn, 0) for fn in fileNameSet]
            hlFea.append(fnVector)
        hlFea_tensor = torch.tensor(hlFea)
        return hlFea_tensor
    
    def getArtifactTypeVector(self, ATSet: Set, edge:List, artfact):
        # 根据edge中的所有节点建立FileNameVector
        hlFea = []
        for node in edge:
            crAt = artfact.getArtifactFeature(artifact_id=node, feature="open_time")
            upAt = artfact.getArtifactFeature(artifact_id=node, feature="updata_time")
            clAt = artfact.getArtifactFeature(artifact_id=node, feature="close_time")
            isPr = artfact.getArtifactFeature(artifact_id=node, feature="is_pr")
            if crAt == None:
                hlFea.append([0]*5)
            if isPr==1:
                type_ = [1, 0]
            else:
                type_ = [0, 1]
            if clAt!=None:
                state = [0, 0 ,1]
            elif upAt!=None:
                state = [0, 1 ,0]
            else:
                state = [1, 0 ,0]
            fnVector = type_ + state
            hlFea.append(fnVector)
        hlFea_tensor = torch.tensor(hlFea)
        return hlFea_tensor

# gFD = generateFeatureDynamically("angular")
# gFD.generateFeatureGivenPair(['10003', '10010'])





