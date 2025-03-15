#!/usr/bin/python
# -*- coding: UTF-8 -*-

import wx

import CalculatorFrame

class CalcFrame(CalculatorFrame.MyFrame1):
    def __init__(self, parent):
        CalculatorFrame.MyFrame1.__init__(self, parent)

    def find_square(self, event):
        num = int(self.m_textCtrl1.GetValue())
        self.m_textCtrl2.SetValue(str(num * num))


app = wx.App(False)

frame = CalcFrame(None)

frame.Show(True)

# start the applications
app.MainLoop()