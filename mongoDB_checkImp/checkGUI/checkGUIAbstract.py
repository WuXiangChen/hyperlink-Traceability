# -*- coding: utf-8 -*-

###########################################################################
## Python code generated with wxFormBuilder (version 4.1.0-0-g733bf3d)
## http://www.wxformbuilder.org/
##
## PLEASE DO *NOT* EDIT THIS FILE!
###########################################################################

import wx
import wx.xrc

TopBar = 1000

###########################################################################
## Class MyFrame2
###########################################################################

class MyFrame2 ( wx.Frame ):

	def __init__( self, parent ):
		wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = wx.EmptyString, pos = wx.DefaultPosition, size = wx.Size( 759,646 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )

		self.SetSizeHints( wx.DefaultSize, wx.DefaultSize )

		bSizer3 = wx.BoxSizer( wx.HORIZONTAL )


		bSizer4 = wx.BoxSizer( wx.VERTICAL )
		self.loadedProject = wx.TextCtrl( self, wx.ID_ANY, u"No Project Loaded", wx.DefaultPosition, wx.DefaultSize, wx.TE_CENTER )
		self.loadedProject.SetMinSize( wx.Size( 200,-1 ) )
		bSizer4.Add( self.loadedProject, 0, wx.ALL|wx.EXPAND, 5 )

		self.m_staticline3 = wx.StaticLine( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.LI_HORIZONTAL )
		bSizer4.Add( self.m_staticline3, 0, wx.ALL|wx.EXPAND, 5 )


		bSizer6 = wx.BoxSizer( wx.HORIZONTAL )
		self.AllNum = wx.StaticText( self, wx.ID_ANY, u"All Figure Num:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.AllNum.Wrap( -1 )

		self.AllNum.SetFont( wx.Font( 13, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, False, wx.EmptyString ) )
		bSizer6.Add( self.AllNum, 0, wx.ALL, 5 )
		self.allNumVariable = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, wx.TE_CENTER )
		bSizer6.Add( self.allNumVariable, 0, 5 )


		bSizer4.Add( bSizer6, 0, wx.ALIGN_CENTER_HORIZONTAL, 5)

		self.m_scrolledWindow1 = wx.ScrolledWindow( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.HSCROLL|wx.VSCROLL )
		self.m_scrolledWindow1.SetScrollRate( 5, 5 )

		bSizer4.Add( self.m_scrolledWindow1, 1, wx.EXPAND, 5 )

		bSizer8 = wx.BoxSizer(wx.HORIZONTAL)

		self.checkAsTure = wx.Button(self, wx.ID_ANY, u"checkAsTrue")
		bSizer8.Add(self.checkAsTure, 0, wx.ALL, 5)

		self.checkAsFalse = wx.Button(self, wx.ID_ANY, u"checkAsFalse")
		bSizer8.Add(self.checkAsFalse, 0, wx.ALL, 5)

		self.checkNotSure = wx.Button(self, wx.ID_ANY, u"NotSure")
		bSizer8.Add(self.checkNotSure, 0, wx.ALL, 5)


		bSizer4.Add( bSizer8, 0, wx.EXPAND, 5 )
		bSizer3.Add( bSizer4, 1,wx.EXPAND, 5 )


		bSizer5 = wx.BoxSizer( wx.VERTICAL )

		self.loadedFigure = wx.TextCtrl( self, wx.ID_ANY, u"No Figures Loaded", wx.DefaultPosition, wx.DefaultSize, wx.TE_CENTER )
		self.loadedFigure.SetMinSize( wx.Size( 200,-1 ) )
		bSizer5.Add( self.loadedFigure, 0, wx.ALL|wx.EXPAND, 5 )

		self.m_staticline4 = wx.StaticLine( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.LI_HORIZONTAL )
		bSizer5.Add( self.m_staticline4, 0, wx.ALL|wx.EXPAND, 5 )

		bSizer7 = wx.BoxSizer( wx.HORIZONTAL )
		self.currNum = wx.StaticText( self, wx.ID_ANY, u"Current Figure Num:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.currNum.Wrap( -1 )

		self.currNum.SetFont( wx.Font( 13, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, False, wx.EmptyString ) )
		bSizer7.Add( self.currNum, 0, wx.ALL, 5 )
		self.currNumVariable = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, wx.TE_CENTER )
		bSizer7.Add( self.currNumVariable, 0, 5 )

		bSizer5.Add( bSizer7, 0, wx.ALIGN_CENTER_HORIZONTAL, 5 )

		self.m_scrolledWindow2 = wx.ScrolledWindow( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.HSCROLL|wx.VSCROLL )
		self.m_scrolledWindow2.SetScrollRate( 5, 5 )
		bSizer5.Add( self.m_scrolledWindow2, 1, wx.EXPAND, 5 )

		bSizer9 = wx.BoxSizer(wx.HORIZONTAL)
		self.leftPanleFind = wx.Button( self, wx.ID_ANY, u"leftFind", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.rightPanleFind = wx.Button(self, wx.ID_ANY, u"rightFind", wx.DefaultPosition, wx.DefaultSize, 0)
		self.FindResult = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, wx.TE_CENTER)
		bSizer9.Add( self.leftPanleFind, 0, wx.ALL, 5 )
		bSizer9.Add(self.rightPanleFind, 0, wx.ALL, 5)
		bSizer9.Add(self.FindResult, 0, wx.ALL, 5)

		bSizer5.Add( bSizer9, 0, wx.EXPAND, 5 )
		bSizer3.Add( bSizer5, 1, wx.EXPAND, 5 )


		# 创建加速表
		accel_tbl = wx.AcceleratorTable([
			(wx.ACCEL_NORMAL, ord('1'), self.checkAsTure.GetId()),
			(wx.ACCEL_NORMAL, ord('2'), self.checkAsFalse.GetId()),
			(wx.ACCEL_NORMAL, ord('3'), self.checkNotSure.GetId()),
			(wx.ACCEL_NORMAL, wx.WXK_LEFT, self.leftPanleFind.GetId()),
			(wx.ACCEL_NORMAL, wx.WXK_RIGHT, self.rightPanleFind.GetId())
		])
		self.SetAcceleratorTable(accel_tbl)


		self.SetSizer( bSizer3 )
		self.Layout()
		self.TopBar = wx.MenuBar( 0 )
		self.Tools = wx.Menu()
		self.loadedFiles = wx.MenuItem( self.Tools, wx.ID_ANY, u"loadedFiles", wx.EmptyString, wx.ITEM_NORMAL )
		self.Tools.Append( self.loadedFiles )

		self.loadedFigures = wx.MenuItem( self.Tools, wx.ID_ANY, u"loadedFigures", wx.EmptyString, wx.ITEM_NORMAL )
		self.Tools.Append( self.loadedFigures )

		self.TopBar.Append( self.Tools, u"Tools" )

		self.STAs = wx.Menu()
		self.statistics = wx.MenuItem( self.STAs, wx.ID_ANY, u"statistics", wx.EmptyString, wx.ITEM_NORMAL )
		self.STAs.Append( self.statistics )

		self.TopBar.Append( self.STAs, u"STAs" )

		self.SetMenuBar( self.TopBar)


		self.Centre( wx.BOTH )

		# Connect Events
		self.checkAsTure.Bind( wx.EVT_BUTTON, self.checkAsTrueEvent )
		self.checkAsFalse.Bind( wx.EVT_BUTTON, self.checkAsFalseEvent )
		self.checkNotSure.Bind(wx.EVT_BUTTON, self.checkNotSureEvent)
		self.leftPanleFind.Bind( wx.EVT_BUTTON, self.leftPanleFindEvent )
		self.rightPanleFind.Bind(wx.EVT_BUTTON, self.rightPanleFindEvent)
		self.Bind( wx.EVT_MENU, self.loadFilesEvent, id = self.loadedFiles.GetId() )
		self.Bind( wx.EVT_MENU, self.loadedFigureEvent, id = self.loadedFigures.GetId() )
		self.Bind( wx.EVT_MENU, self.ShowSTAsEvent, id = self.statistics.GetId() )

	def __del__( self ):
		pass


	# Virtual event handlers, override them in your derived class
	def checkAsTrueEvent( self, event ):
		event.Skip()

	def checkAsFalseEvent(self, event ):
		event.Skip()
	def checkNotSureEvent(self, event ):
		event.Skip()

	def leftPanleFindEvent( self, event ):
		event.Skip()

	def rightPanleFindEvent(self, event):
		event.Skip()

	def loadFilesEvent( self, event ):
		event.Skip()

	def loadedFigureEvent( self, event ):
		event.Skip()

	def ShowSTAsEvent( self, event ):
		event.Skip()


