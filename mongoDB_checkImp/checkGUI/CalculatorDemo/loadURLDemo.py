import wx
import wx.html2

class MyFrame(wx.Frame):
    def __init__(self, parent):
        super(MyFrame, self).__init__(parent, title="WebView in Scrolled Window")

        self.m_scrolledWindow1 = wx.ScrolledWindow(self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.HSCROLL|wx.VSCROLL)
        self.m_scrolledWindow1.SetScrollRate(5, 5)

        # 创建两个 WebView 子控件
        webview1 = wx.html2.WebView.New(self.m_scrolledWindow1)
        webview1.LoadURL("https://www.example.com")

        webview2 = wx.html2.WebView.New(self.m_scrolledWindow1)
        webview2.LoadURL("https://www.google.com")
        webview2.Hide()  # 初始隐藏第二个 WebView

        self.btn_update = wx.Button(self.m_scrolledWindow1, label="Switch WebView")
        self.Bind(wx.EVT_BUTTON, self.on_update_webview, self.btn_update)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(webview1, 1, wx.EXPAND)
        sizer.Add(webview2, 1, wx.EXPAND)
        sizer.Add(self.btn_update, 0, wx.ALIGN_CENTER | wx.TOP, 10)
        self.m_scrolledWindow1.SetSizer(sizer)

        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(self.m_scrolledWindow1, 1, wx.EXPAND)
        self.SetSizer(main_sizer)

        self.current_webview = webview1  # 记录当前显示的 WebView

    def on_update_webview(self, event):
        # 切换当前显示的 WebView
        child_control = self.m_scrolledWindow1.GetChildren()[0]
        child_control.Destroy()

        new_webview = self.m_scrolledWindow1.GetChildren()[0]
        new_webview.Show()
        # 重新设置m_scrolledWindow1的尺寸
        self.m_scrolledWindow1.Layout()


if __name__ == '__main__':
    app = wx.App()
    frame = MyFrame(None)
    frame.Show()
    app.MainLoop()
