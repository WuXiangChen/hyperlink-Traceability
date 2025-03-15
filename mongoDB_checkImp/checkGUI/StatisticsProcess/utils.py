# 本节提供一些基本的操作
import simplejson as json
import os

def readJsonFile(filePath):
    with open(filePath, 'r') as f:
        data = json.load(f)
    return data


def writeJsonFile(filePath, data):
    with open(filePath, 'w') as f:
        json.dump(data, f)

def get_file_order(filePath):
    """
    获取一个文件在其所在文件夹中的数量排序
    :param file_path: 文件的完整路径
    :return: 该文件在文件夹中的数量排序
    """
    # 获取该目录下所有文件的文件名
    folderPath = os.path.dirname(filePath)
    fileName = os.path.basename(filePath)

    fileLS = os.listdir(folderPath)
    fileLS.sort(key=lambda x: int(x[:-5]))
    lenLS = len(fileLS)
    try:
        return lenLS, fileLS.index(fileName) + 1
    except ValueError:
        return lenLS, -1

def getNextFilePath(filePath,currentNum):
    # 获取该目录下所有文件的文件名
    folderPath = os.path.dirname(filePath)

    fileLS = os.listdir(folderPath)
    fileLS.sort(key=lambda x: int(x[:-5]))
    lenLS = len(fileLS)
    if currentNum == lenLS:
        return None
    else:
        return os.path.join(folderPath,fileLS[currentNum])


def get_original_figures_path(rootOriginalFiguresPath, owner, repo, allFigureFolders):
    """
    Find the original figures path based on the given owner and repository.

    Args:
        rootOriginalFiguresPath (str): The root path of the original figures.
        owner (str): The owner of the repository.
        repo (str): The name of the repository.
        allFigureFolders (list): A list of all figure folders.

    Returns:
        str: The path to the original figures, or None if not found.
    """
    for figureFolder in allFigureFolders:
        if owner in figureFolder and repo in figureFolder:
            originalFiguresPath = os.path.join(rootOriginalFiguresPath, figureFolder)
            return originalFiguresPath
    return None

def printSTAs(correctRatio, falseRatio, notSureRatio):
    print(f"Correct Ratio by linking: {correctRatio}")
    print(f"False Ratio by linking: {falseRatio}")
    print(f"Not Sure Ratio by linking: {notSureRatio}")
    print("=============================")