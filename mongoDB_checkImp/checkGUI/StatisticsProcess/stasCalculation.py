# 本节的主要目的是对已verified Linking Results进行统计并完成绘图展示
# 本节所需的数据 基本上都依赖于 “checkGUI/DataLayer/DataPersistenceVerified”中的实时记录结果

import os
from collections import defaultdict

import numpy as np
from glob import glob
from checkGUI.StatisticsProcess.utils import readJsonFile, get_original_figures_path,printSTAs
import pandas as pd

def correctRationByLinking(linkDF):
    # 按照链接 计算正确的比例
    correctRatio = linkDF[linkDF['VerifiedState'] == 1].shape[0] / linkDF.shape[0]
    falseRatio = linkDF[linkDF['VerifiedState'] == 0].shape[0] / linkDF.shape[0]
    notSureRatio = linkDF[linkDF['VerifiedState'] == -1].shape[0] / linkDF.shape[0]
    return [correctRatio, falseRatio, notSureRatio]

def correctRationByGraph(graphVerified):
    if len(graphVerified) == 0:
        return [0, 0, 0]
    # 将graphVerified修正为合适的格式
    tempDict = defaultdict(int)
    for key, value in graphVerified.items():
        if -1 in value.values() or 0 in value.values():
            tempDict[key] = 0
        else:
            tempDict[key] = 1
    tempDF = pd.DataFrame.from_dict(tempDict, orient='index')
    tempDF.columns = ['VerifiedState']
    correctRatio, falseRatio, notSureRatio = correctRationByLinking(tempDF)

    return [correctRatio, falseRatio, notSureRatio]

def deriveLinkingFromGraph(graphVerified):
    # 从图中提取出链接和其状态
    linkingDict = defaultdict(dict)
    for key, value in graphVerified.items():
        for k,v in value.items():
            if k == "AllNodes":
                continue
            else:
                linkingDict.update({k: v})
    linkingDF = pd.DataFrame.from_dict(linkingDict, orient='index')
    linkingDF.columns = ['VerifiedState']
    return linkingDF


def correctRationByLinking_Type(graphVerified):
    # 按照链接和类型计算正确的比例
    # 先分大小图
    smallDict, biggerDict = splitGraphByLinkingNum(graphVerified)
    smallDF = deriveLinkingFromGraph(smallDict)
    biggerDF = deriveLinkingFromGraph(biggerDict)
    smallPart = correctRationByLinking(smallDF)
    bigPart = correctRationByLinking(biggerDF)
    return smallPart, bigPart

def correctRationSeparateByGraphAndKeyWord_TypeBasedOnLinking(graphVerified,graphVerifiedWeight):
    # 按照链接和类型计算正确的比例
    # 先分大小图
    smallDict, biggerDict = splitGraphByLinkingNum(graphVerified)
    # 再按照Keyword比例区分是否是关键图
    smallKeyDict, smallNoKeyDict = separateLinkingByKeyWord(smallDict,graphVerifiedWeight)
    bigKeyDict, bigNoKeyDict = separateLinkingByKeyWord(biggerDict,graphVerifiedWeight)

    smallKeyDF = deriveLinkingFromGraph(smallKeyDict)
    smallNoKeyDF = deriveLinkingFromGraph(smallNoKeyDict)
    bigKeyDF = deriveLinkingFromGraph(bigKeyDict)
    bigNoKeyDF = deriveLinkingFromGraph(bigNoKeyDict)

    smallKeyPart = correctRationByLinking(smallKeyDF)
    smallNoKeyPart = correctRationByLinking(smallNoKeyDF)
    bigKeyPart = correctRationByLinking(bigKeyDF)
    bigNoKeyPart = correctRationByLinking(bigNoKeyDF)
    return smallKeyPart, smallNoKeyPart, bigKeyPart, bigNoKeyPart

def splitGraphByLinkingNum(graphVerified):
    #先分再合
    smallDict = defaultdict(dict)
    biggerDict = defaultdict(dict)
    for key, value in graphVerified.items():
        if len(value) > 10:
            biggerDict[key] = value
        else:
            smallDict[key] = value
    return smallDict, biggerDict


def correctRationByGraph_Type(graphVerified):
    smallDict, biggerDict = splitGraphByLinkingNum(graphVerified)
    smallPart = correctRationByGraph(smallDict)
    bigPart = correctRationByGraph(biggerDict)
    return smallPart, bigPart

def separateLinkingByKeyWord(graphVerified, graphVerifiedWeight):
    keywordPart, NokeywordPart = defaultdict(dict), defaultdict(dict)
    for key, value in graphVerifiedWeight.items():
        if key not in graphVerified.keys():
            continue
        keyCon = value.keys()
        for con in keyCon:
            if value[con] == 100000000:
                    keywordPart[key].update({con: graphVerified[key][con]})
            else:
                    NokeywordPart[key].update({con: graphVerified[key][con]})
    return keywordPart, NokeywordPart

def separateGraphByKeyWord(smallDict, graphVerifiedWeight):
    keyGraph, noKeyGraph = defaultdict(dict), defaultdict(dict)

    for key, value in smallDict.items():
        if key not in graphVerifiedWeight.keys():
            continue
        counter = 0
        keyCon = value.keys()
        sumCounter = len(keyCon)-1
        for con in keyCon:
            if con=="AllNodes":
                continue
            if graphVerifiedWeight[key][con] == 100000000:
                counter+=1
        if counter/sumCounter > 0.5:
            keyGraph[key] = value
        else:
            noKeyGraph[key] = value

    return keyGraph, noKeyGraph

def correctRationSeparateByGraphAndKeyWord_TypeBasedOnGraph(graphVerified,graphVerifiedWeight):
    smallDict, biggerDict = splitGraphByLinkingNum(graphVerified)
    # 再按照Keyword比例区分是否是关键图
    smallKeyDict, smallNoKeyDict = separateGraphByKeyWord(smallDict, graphVerifiedWeight)
    bigKeyDict, bigNoKeyDict = separateGraphByKeyWord(biggerDict, graphVerifiedWeight)
    smallKeyPart = correctRationByGraph(smallKeyDict)
    smallNoKeyPart = correctRationByGraph(smallNoKeyDict)

    bigKeyPart = correctRationByGraph(bigKeyDict)
    bigNoKeyPart = correctRationByGraph(bigNoKeyDict)
    return smallKeyPart, smallNoKeyPart, bigKeyPart, bigNoKeyPart

def readCompleteGraphs(conComponent, linkDF):
    # 读取所有的完整图, 并将验证结果进行赋值
    # 最终的返回结果将是一个字典，key是子图，value是{linking:VerifiedState}
    graphVerified = defaultdict(dict)
    subgraphs = set()
    allgraphs = set()

    print("KeyWord type graph generated, and some linking not recognized as true:")
    for i,subGraph in enumerate(conComponent):
        flag = True
        if i == 2:
            print("Debug")
        if len(subGraph) < 2:
            continue
        allgraphs.add(i)
        rawName = linkDF.index
        for nodeName in rawName:
            leftNode, rightNode = nodeName.split("-")[0],nodeName.split("-")[1]
            if leftNode in subGraph and rightNode in subGraph:
                if flag == True:
                    flag = False
                    graphVerified[i] = {"AllNodes": subGraph}
                    subgraphs.add(i)
                graphVerified[i].update({nodeName: linkDF.loc[nodeName, 'VerifiedState']})

            if leftNode in subGraph and rightNode not in subGraph:
                print(f"Node Not Found: {rightNode}-{leftNode}")
            if leftNode not in subGraph and rightNode in subGraph:
                print(f"Node Not Found: {leftNode}-{rightNode}")
    diffgraph = allgraphs - subgraphs
    return graphVerified

def generateCompleteGraphsWeight(oriFigureFolderPath, graphVerified):
    # 读取所有的完整图, 并将验证结果进行赋值
    # 最终的返回结果将是一个字典，key是子图，value是{linking:VerifiedState}
    weakerLabelingPath = os.path.join(oriFigureFolderPath, "weakerLabeling.json")
    weakerLabeling = readJsonFile(weakerLabelingPath)

    weakerLabelingWeightPath = os.path.join(oriFigureFolderPath, "weakerLabelingWeight.json")
    weakerLabelingWeight = readJsonFile(weakerLabelingWeightPath)

    graphVerifiedWeight = defaultdict(dict)
    allgraphs = set()
    for i,figNum in enumerate(graphVerified):
        allgraphs.add(figNum)
        keys = graphVerified[figNum].keys()
        for nodeName in keys:
            if nodeName == "AllNodes":
                continue
            leftNode, rightNode = nodeName.split("-")[0],nodeName.split("-")[1]
            if leftNode in weakerLabeling.keys() and rightNode in weakerLabeling[leftNode]:
                Index = weakerLabeling[leftNode].index(rightNode)
                tempWeight = weakerLabelingWeight[leftNode][Index]
            else:
                Index = weakerLabeling[rightNode].index(leftNode)
                tempWeight = weakerLabelingWeight[rightNode][Index]
            graphVerifiedWeight[figNum].update({nodeName: tempWeight})
    return graphVerifiedWeight

def judgeComplete(graphVerified):
    # 判断是否已经填满
    notComplete = []
    for key,value in graphVerified.items():
        allNodes = set(value['AllNodes'])
        lenNode = len(allNodes)
        keys = list(value.keys())
        keys.remove('AllNodes')
        Nodes = set()

        for k in keys:
            leftNode, rightNode = k.split("-")[0], k.split("-")[1]
            Nodes.add(leftNode)
            Nodes.add(rightNode)

        if len(Nodes) != lenNode:
            diffNodes = allNodes - Nodes
            notComplete.append({key: diffNodes})
            continue
    return notComplete

def correctRationByKeyWord(graphVerified, graphVerifiedWeight):
    keywordPart, NokeywordPart = separateLinkingByKeyWord(graphVerified, graphVerifiedWeight)
    keywordDF = deriveLinkingFromGraph(keywordPart)
    NokeywordDF = deriveLinkingFromGraph(NokeywordPart)
    keywordLink = correctRationByLinking(keywordDF)
    NokeywordLink = correctRationByLinking(NokeywordDF)
    return keywordLink, NokeywordLink

def main(allFiles):
    for relFilePath in allFiles:
        print(f"Processing File: {relFilePath}")
        # 解析项目的owner和repo
        fileName = os.path.basename(relFilePath)
        owner, repo = fileName.split("_")[0:2]

        # 加载验证结果集
        linkJson = readJsonFile(relFilePath)
        linkDF = pd.DataFrame.from_dict(linkJson, orient='index')
        linkDF.columns = ['VerifiedState']
        del linkJson

        # 加载原始图集，根据owner和repo进行匹配检索
        oriFigureFolderPath = get_original_figures_path(rootOriginalFiguresPath, owner, repo, allFigureFolders)
        conPath = os.path.join(oriFigureFolderPath, "connectedComponent.json")
        conComponent = readJsonFile(conPath)

        if oriFigureFolderPath is None:
            print(f"Original Figure Folder Not Found: {owner}_{repo}")
            continue

        # 将LinkDF和conComponent进行合并
        graphVerified = readCompleteGraphs(conComponent, linkDF)
        # 接下来对合并图进行判满，统计是否linking 结果已经填满所选图
        NotCompleted = judgeComplete(graphVerified)
        notGraphNum = list(list(key.keys())[0] for key in  NotCompleted)
        graphVerifiedWeight = generateCompleteGraphsWeight(oriFigureFolderPath,graphVerified)

        # 弹出未满子图
        for index in notGraphNum:
            graphVerified.pop(index)
            graphVerifiedWeight.pop(index)
        print("图未满列表：", NotCompleted)

        print("+"*80)
        # 总图数目，总linking统计
        print(f"总图数目：{len(graphVerified)}")
        print(f"总Linking数目：{linkDF.shape[0]}")

        # 进行Linking统计
        correctRatio, falseRatio, notSureRatio = correctRationByLinking(linkDF)
        print(f"==========统计类型：Links============")
        printSTAs(correctRatio, falseRatio, notSureRatio)

        smallPartLink, biggerPartLink = correctRationByLinking_Type(graphVerified)
        print(f"==========统计类型：Links-Type-Small============")
        printSTAs(smallPartLink[0], smallPartLink[1], smallPartLink[2])
        print(f"==========统计类型：Links-Type-Bigger============")
        printSTAs(biggerPartLink[0], biggerPartLink[1], biggerPartLink[2])

        # 进行图统计
        keywordLink, NokeywordLink = correctRationByKeyWord(graphVerified, graphVerifiedWeight)
        print(f"==========统计类型：keyword Link============")
        printSTAs(keywordLink[0], keywordLink[1], keywordLink[2])
        print(f"==========统计类型：No keyword Link============")
        printSTAs(NokeywordLink[0], NokeywordLink[1], NokeywordLink[2])

        correctRatio, falseRatio, notSureRatio = correctRationByGraph(graphVerified)
        print(f"==========统计类型：Graphs============")
        printSTAs(correctRatio, falseRatio, notSureRatio)

        smallPartGraph, biggerPartGraph = correctRationByGraph_Type(graphVerified)
        print(f"==========统计类型：Graphs-Type-Small============")
        printSTAs(smallPartGraph[0], smallPartGraph[1], smallPartGraph[2])
        print(f"==========统计类型：Graphs-Type-Bigger============")
        printSTAs(biggerPartGraph[0], biggerPartGraph[1], biggerPartGraph[2])

        # 这边还得进行细分，细分的规则是按照是否包含keyword标记进行区分
        smallKeyPart, smallNoKeyPart, bigKeyPart, bigNoKeyPart =\
            correctRationSeparateByGraphAndKeyWord_TypeBasedOnLinking(graphVerified,graphVerifiedWeight)

        print(f"==========统计类型：Links-Type-Small-Keyword============")
        printSTAs(smallKeyPart[0], smallKeyPart[1], smallKeyPart[2])
        print(f"==========统计类型：Links-Type-Small-NoKeyword============")
        printSTAs(smallNoKeyPart[0], smallNoKeyPart[1], smallNoKeyPart[2])
        print(f"==========统计类型：Links-Type-Bigger-Keyword============")
        printSTAs(bigKeyPart[0], bigKeyPart[1], bigKeyPart[2])
        print(f"==========统计类型：Links-Type-Bigger-NoKeyword============")
        printSTAs(bigNoKeyPart[0], bigNoKeyPart[1], bigNoKeyPart[2])

        smallKeyPart, smallNoKeyPart, bigKeyPart, bigNoKeyPart =\
            correctRationSeparateByGraphAndKeyWord_TypeBasedOnGraph(graphVerified,graphVerifiedWeight)
        #smallPartLink, biggerPartLink = correctRationByLinking_Type(graphVerified)
        print(f"==========统计类型：Graphs-Type-Small-Keyword============")
        printSTAs(smallKeyPart[0], smallKeyPart[1], smallKeyPart[2])
        print(f"==========统计类型：Graphs-Type-Small-NoKeyword============")
        printSTAs(smallNoKeyPart[0], smallNoKeyPart[1], smallNoKeyPart[2])
        print(f"==========统计类型：Graphs-Type-Bigger-Keyword============")
        printSTAs(bigKeyPart[0], bigKeyPart[1], bigKeyPart[2])
        print(f"==========统计类型：Graphs-Type-Bigger-NoKeyword============")
        printSTAs(bigNoKeyPart[0], bigNoKeyPart[1], bigNoKeyPart[2])


        print("+"*80)

if __name__ == '__main__':
    rootFolderPath = os.path.relpath("../DataLayer/DataPersistenceVerified/*.json")
    rootOriginalFiguresPath = os.path.relpath(
        "../../GenerateDataset/generateWeakLabeler/weakerLabeledPlusKeyWordResults/")
    # 读取所有的文件夹
    allFiles = glob(rootFolderPath)[0]
    allFigureFolders = os.listdir(rootOriginalFiguresPath)
    main([allFiles])

