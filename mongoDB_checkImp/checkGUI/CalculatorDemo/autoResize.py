#!/usr/bin/python
# -*- coding: UTF-8 -*-

import wx

import autoResizeFrame

class CalcFrame(autoResizeFrame.MyFrame2):
    def __init__(self, parent):
        autoResizeFrame.MyFrame2.__init__(self, parent)


app = wx.App(False)

frame = CalcFrame(None)

frame.Show(True)

# start the applications
app.MainLoop()