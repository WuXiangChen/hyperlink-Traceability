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
        # Get the letter to search for
        letter = "a"  # You can change this to the desired letter

        # Run the JavaScript code to find the first occurrence of the letter
        a = self.webview.RunScript(f"""
        var content = document.body.innerText;
        var index = content.indexOf('{letter}');
        if (index !== -1) {{
            var element = document.elementFromPoint(0, index);
            var rect = element.getBoundingClientRect();
            window.parent.postMessage({{
                type: 'scroll_position',
                scrollY: rect.top
            }}, '*');
        }} else {{
            window.parent.postMessage({{
                type: 'not_found'
            }}, '*');
        }}
        """)

        # Listen for the message from the JavaScript code
        self.Bind(wx.html2.EVT_WEBVIEW_LOADED, self.on_webview_loaded, self.webview)

    def on_webview_loaded(self, event):
        letter = "a"  # You can change this to the desired letter
        def on_message(evt):
            data = evt.GetMessage()
            if data['type'] == 'scroll_position':
                scroll_position = int(data['scrollY'])
                print(f"The letter is at position {scroll_position}")
                # You can now use this scroll_position value to scroll the m_scrolledWindow
                self.m_scrolledWindow.Scroll(0, scroll_position)
            elif data['type'] == 'not_found':
                wx.MessageBox(f"'{letter}' not found in the page.", "Message", wx.OK | wx.ICON_INFORMATION)
        self.webview.Bind(wx.html2.EVT_WEBVIEW_MESSAGE, on_message)

if __name__ == "__main__":
    app = wx.App()
    frame = MyFrame(None)
    frame.Show()
    app.MainLoop()