import wx
import wx.html2

class MyFrame(wx.Frame):
    def __init__(self, parent):
        super().__init__(parent, title="My Frame")

        # Create the scroll window
        self.m_scrolledWindow = wx.ScrolledWindow(self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.VSCROLL)
        self.m_scrolledWindow.SetScrollRate(20, 20)

        # Create the WebView
        self.webview = wx.html2.WebView.New(self.m_scrolledWindow)
        self.webview.LoadURL("https://www.example.com")

        box_sizer = wx.BoxSizer(wx.VERTICAL)
        box_sizer.Add(self.webview, 1, wx.EXPAND)
        self.m_scrolledWindow.SetSizer(box_sizer)

        # Add a button to find the first occurrence of a letter
        self.find_button = wx.Button(self, wx.ID_ANY, u"Find Letter", wx.DefaultPosition, wx.DefaultSize, 0)
        self.Bind(wx.EVT_BUTTON, self.on_find_button_click, self.find_button)

        # Layout the controls
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(self.m_scrolledWindow, 1, wx.EXPAND | wx.ALL, 5)
        main_sizer.Add(self.find_button, 0, wx.ALIGN_CENTER | wx.TOP, 5)
        self.SetSizer(main_sizer)
        self.Layout()

    def on_find_button_click(self, event):
        search_text = "permission"
        self.webview.Find(search_text)

if __name__ == "__main__":
    app = wx.App()
    frame = MyFrame(None)
    frame.Show()
    app.MainLoop()
