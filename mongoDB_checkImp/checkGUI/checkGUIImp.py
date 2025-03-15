#!/usr/bin/python
# -*- coding: UTF-8 -*-
import wx
import checkGUIAbstract
import os
import wx.html2
from DataLayer.DataManipulateLayer import DataManipulateLayer
from DataLayer.FiguresR import FiguresR
from checkGUI.GUI_utils import token

# from WebSiteViewer import webViewer

# 这里进行整改，不再以文件为基本单位，改成以文件中记载的图像为基本单位
# 以此 才能实现连续加载图像的功能
class CalcFrame(checkGUIAbstract.MyFrame2):
    def __init__(self, parent):
        checkGUIAbstract.MyFrame2.__init__(self, parent)
        self.folder_path = None
        # 这里是数据持久化的结果
        self.dml = None
        self.figs = None
        self.currentNode = None
        self.nextNodes = None
        # 预加载网页信息
        self.currentNum = 0
        self.allLinkingNum = 0

    def handleLinkingEvent(self, event, is_true_event):
        """
        处理链接事件,记录链接信息并处理下一步操作。
        Args:
            event: 事件对象
            is_true_event: 布尔值(-1,0,1),表示是否为正确的链接事件。
        """
        leftNode,rightNode = self.currentNode
        self.dml.recordLinking(leftNode,rightNode,is_true_event)
        print(f"Recorded: {leftNode} and {rightNode} as {is_true_event}")
        if self.nextNodes is not None:
            self.currentNode = self.nextNodes
            self.currNumVariable.SetValue(str(self.currentNum))
            self.ShowNextElementsEvent()

        self.preLoadedFigure()
        if self.nextNodes is not None:
            self.preLoadNextWebPage(self.nextNodes[0], self.nextNodes[1])
        else:
            self.loadedFigure.SetValue("All files have been loaded")
            return None


    def checkAsTrueEvent(self, event):
        """
        处理正确的链接事件。
        """
        self.handleLinkingEvent(event, 1)

    def checkAsFalseEvent(self, event):
        """
        处理错误的链接事件。
        """
        self.handleLinkingEvent(event, 0)

    def checkNotSureEvent(self, event):
        self.handleLinkingEvent(event, -1)

    # 这里要和数据库进行交互，获取下一个未被判断的Linking
    def leftPanleFindEvent(self, event):
        curLeftNode, curRightNode = self.currentNode
        a = self.m_scrolledWindow1.GetChildren()[0].Find(curRightNode)
        if a == -1:
            self.FindResult.SetValue("Not Found")
        else:
            self.FindResult.SetValue("Found")

    def rightPanleFindEvent(self, event):
        curLeftNode, curRightNode = self.currentNode
        a = self.m_scrolledWindow2.GetChildren()[0].Find(curLeftNode)
        if a == -1:
            self.FindResult.SetValue("Not Found")
        else:
            self.FindResult.SetValue("Found")

    def loadFilesEvent(self, event):
        dir_dialog = wx.DirDialog(self, "Choose a directory", style=wx.DD_DEFAULT_STYLE)
        if dir_dialog.ShowModal() == wx.ID_OK:
            self.folder_path = dir_dialog.GetPath()
            print(f"Selected folder: {self.folder_path}")
        dir_dialog.Destroy()
        # 这里按照所选项目的类型进行分类处理
        if self.folder_path is None:
            self.loadedProject.SetValue("No folder selected")
        else:
            self.selectedProject = self.folder_path.split("\\")[-2]
            self.loadedProject.SetValue(self.selectedProject)
            # 获取 owner和repo
            owner = self.selectedProject.split("_")[1]
            repo = self.selectedProject.split("_")[0]
            # 在选择了项目的时候，就可以定义处理持久化的数据部分了
            self.dml = DataManipulateLayer(owner, repo)
            # 这里可以直接用folder_path初始化figures
            self.figs = FiguresR(self.folder_path,self.dml)
            self.initialSelectFigure()

    def initialSelectFigure(self):
        self.currentNum += 1
        self.currentNode = self.figs.getFigure()
        self.allLinkingNum = self.figs.figures.qsize()
        self.updateBar()
        self.ShowFirstElementsEvent(first=True)

    def loadedFigureEvent(self, event):
        return event

    def ShowFirstElementsEvent(self,first=False):
        while self.currentNode is None:
            self.loadedFigure.SetValue("All files have been loaded")
            return None

        leftNode, rightNode = self.currentNode
        self.preLoadNextWebPage(leftNode, rightNode)
        self.ShowNextElementsEvent(first=first)
        self.preLoadedFigure()
        # 找到下一个节点，预加载下一个网页
        if self.nextNodes is not None:
            leftNode, rightNode = self.nextNodes
            self.preLoadNextWebPage(leftNode, rightNode)

    def clearAndDestroyFirstPage(self):
        if self.m_scrolledWindow1.GetChildren()[0] is not None:
            self.m_scrolledWindow1.GetChildren()[0].Hide()
            self.m_scrolledWindow1.GetChildren()[0].Destroy()

        if self.m_scrolledWindow2.GetChildren()[0] is not None:
            self.m_scrolledWindow2.GetChildren()[0].Hide()
            self.m_scrolledWindow2.GetChildren()[0].Destroy()

    def ShowNextElementsEvent(self,first=False):
        if not first:
            self.clearAndDestroyFirstPage()
        self.m_scrolledWindow1.GetChildren()[0].Show()
        self.m_scrolledWindow2.GetChildren()[0].Show()
        # 再调整布局
        self.m_scrolledWindow1.Layout()
        self.m_scrolledWindow2.Layout()

    def preLoadNextWebPage(self, leftNode, rightNode):
        leftPanleview = wx.html2.WebView.New(self.m_scrolledWindow1)
        leftpageViewage = "https://github.com/{}/{}/issues/{}?access_token={}".format(self.dml.owner, self.dml.repo, leftNode,"token "+token)
        leftPanleview.LoadURL(leftpageViewage)

        rightPanleview = wx.html2.WebView.New(self.m_scrolledWindow2)
        rightpageViewage = "https://github.com/{}/{}/issues/{}?access_token={}".format(self.dml.owner, self.dml.repo, rightNode,"token "+token)
        rightPanleview.LoadURL(rightpageViewage)

        sizerLeft = wx.BoxSizer(wx.VERTICAL)
        sizerLeft.Add(leftPanleview, 1, wx.EXPAND)

        sizerRight = wx.BoxSizer(wx.VERTICAL)
        sizerRight.Add(rightPanleview, 1, wx.EXPAND)

        self.m_scrolledWindow1.SetSizer(sizerLeft)
        self.m_scrolledWindow2.SetSizer(sizerRight)

    def updateBar(self):
        self.allNumVariable.SetValue(str(self.allLinkingNum))
        self.currNumVariable.SetValue(str(self.currentNum))

    def preLoadedFigure(self):
        self.nextNodes = self.figs.getFigure()
        self.currentNum += 1


app = wx.App(False)

frame = CalcFrame(None)

frame.Show(True)

# start the applications
app.MainLoop()