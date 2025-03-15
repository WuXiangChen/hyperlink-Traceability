# -*- coding: utf-8 -*-

###########################################################################
## Python code generated with wxFormBuilder (version 4.1.0-0-g733bf3d)
## http://www.wxformbuilder.org/
##
## PLEASE DO *NOT* EDIT THIS FILE!
###########################################################################

import wx
import wx.xrc

###########################################################################
## Class MyFrame1
###########################################################################

class MyFrame1 ( wx.Frame ):

	def __init__( self, parent ):
		wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = wx.EmptyString, pos = wx.DefaultPosition, size = wx.Size( 840,561 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )

		self.SetSizeHints( wx.DefaultSize, wx.DefaultSize )

		m_label_num = wx.BoxSizer( wx.VERTICAL )

		self.m_label_num = wx.StaticText( self, wx.ID_ANY, u"请输入一个数字", wx.DefaultPosition, wx.Size( 900,-1 ), wx.ALIGN_CENTER_HORIZONTAL )
		self.m_label_num.Wrap( -1 )

		m_label_num.Add( self.m_label_num, 0, wx.ALL, 5 )

		self.m_textCtrl1 = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( 900,-1 ), wx.TE_CENTER )
		m_label_num.Add( self.m_textCtrl1, 0, wx.ALL, 5 )

		self.m_button1 = wx.Button( self, wx.ID_ANY, u"求该数字的平方", wx.DefaultPosition, wx.Size( 900,-1 ), wx.BU_BOTTOM )
		m_label_num.Add( self.m_button1, 0, wx.ALL, 5 )

		self.m_textCtrl2 = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( 900,-1 ), wx.TE_CENTER )
		m_label_num.Add( self.m_textCtrl2, 0, wx.ALL, 5 )


		self.SetSizer( m_label_num )
		self.Layout()

		self.Centre( wx.BOTH )

		# Connect Events
		self.m_button1.Bind( wx.EVT_BUTTON, self.find_square )

	def __del__( self ):
		pass


	# Virtual event handlers, override them in your derived class
	def find_square( self, event ):
		event.Skip()


