# 加载网页 并进行显示
import wx
import os
import wx.html2

class webViewer(wx.Frame):
    def __init__(self, dml, m_scrolledWindow1, m_scrolledWindow2):
        super(webViewer, self).__init__()
        self.dml = dml
        self.m_scrolledWindow1 = m_scrolledWindow1
        self.m_scrolledWindow2 = m_scrolledWindow2

        self.left_content_sizer = None
        self.right_content_sizer = None


    def loadWeb(self, leftNode,rightNode):

        if leftNode is None or rightNode is None:
            raise ValueError("WebPage: Please input the leftNode and rightNode!")

        self.leftPanleview = wx.html2.WebView.New()
        leftWebPage = "https://github.com/{}/{}/issues/{}".format(self.dml.owner, self.dml.repo, leftNode)
        self.leftPanleview.LoadURL(leftWebPage)
        self.left_content_sizer = wx.BoxSizer(wx.VERTICAL)
        self.left_content_sizer.Add(self.leftPanleview, 1, wx.EXPAND)


        self.rightPanleview = wx.html2.WebView.New()
        rightWebPage = "https://github.com/{}/{}/issues/{}".format(self.dml.owner, self.dml.repo, rightNode)
        self.rightPanleview.LoadURL(rightWebPage)
        self.right_content_sizer = wx.BoxSizer(wx.VERTICAL)
        self.right_content_sizer.Add(self.rightPanleview, 1, wx.EXPAND)


    def showWeb(self):
        if self.left_content_sizer is None or self.right_content_sizer is None:
           raise ValueError("Please load the web page first!")

        self.m_scrolledWindow1.SetSizer(self.left_content_sizer)
        # Set the virtual size of the scrolled window
        self.m_scrolledWindow1.SetVirtualSize(self.rightPanleview.GetSize())

        self.m_scrolledWindow2.SetSizer(self.right_content_sizer)
        # Set the virtual size of the scrolled window
        self.m_scrolledWindow2.SetVirtualSize(self.rightPanleview.GetSize())