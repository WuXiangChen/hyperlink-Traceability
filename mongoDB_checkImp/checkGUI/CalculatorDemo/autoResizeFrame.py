import wx

class MyFrame(wx.Frame):
    def __init__(self, parent):
        wx.Frame.__init__(self, parent, id=wx.ID_ANY, title="My Frame", size=(800, 600))

        self.panel = MyPanel(self)
        self.Bind(wx.EVT_SIZE, self.on_resize, self)

    def on_resize(self, event):
        self.panel.Layout()
        event.Skip()

class MyPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, id=wx.ID_ANY)

        # Create the initializedInfo sizer
        main_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # Create the left sizer
        left_sizer = wx.BoxSizer(wx.VERTICAL)

        # Add components to the left sizer
        self.left_text = wx.StaticText(self, label="Left Side")
        left_sizer.Add(self.left_text, 0, wx.ALL | wx.ALIGN_CENTER, 10)

        self.left_list = wx.ListCtrl(self, style=wx.LC_REPORT)
        self.left_list.InsertColumn(0, "Item")

        for i in range(10):
            self.left_list.Append([f"Item {i+1}"])

        left_sizer.Add(self.left_list, 1, wx.ALL | wx.EXPAND, 10)

        # Create the right sizer
        right_sizer = wx.BoxSizer(wx.VERTICAL)

        # Add components to the right sizer
        self.right_text = wx.StaticText(self, label="Right Side")
        right_sizer.Add(self.right_text, 0, wx.ALL | wx.ALIGN_CENTER, 10)

        self.right_list = wx.ListCtrl(self, style=wx.LC_REPORT)
        self.right_list.InsertColumn(0, "Item")
        for i in range(10):
            self.right_list.Append([f"Item {i+1}"])
        right_sizer.Add(self.right_list, 1, wx.ALL | wx.EXPAND, 10)

        # Add the left and right sizers to the initializedInfo sizer
        main_sizer.Add(left_sizer, 1, wx.EXPAND)
        main_sizer.Add(right_sizer, 1, wx.EXPAND)

        self.SetSizer(main_sizer)

if __name__ == '__main__':

    app = wx.App()
    frame = MyFrame(None)
    frame.Show()
    app.MainLoop()
