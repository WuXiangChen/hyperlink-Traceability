import wx

class MyFrame(wx.Frame):
    def __init__(self, parent):
        wx.Frame.__init__(self, parent, id=wx.ID_ANY, title="Webpage in Scrolled Window")

        # Create the initializedInfo panel with a scrolled window
        self.panel = wx.Panel(self)
        self.scrolled_window = wx.ScrolledWindow(self.panel)

        # Create a sizer for the panel and add the scrolled window
        panel_sizer = wx.BoxSizer(wx.VERTICAL)
        panel_sizer.Add(self.scrolled_window, 1, wx.EXPAND)

        # Add a button to load files
        self.load_files_button = wx.Button(self.panel, label="Load Files")
        self.Bind(wx.EVT_BUTTON, self.load_files_event, self.load_files_button)
        panel_sizer.Add(self.load_files_button, 0, wx.TOP | wx.ALIGN_RIGHT, 10)

        self.panel.SetSizer(panel_sizer)

        # Create a sizer for the frame and add the panel
        frame_sizer = wx.BoxSizer(wx.VERTICAL)
        frame_sizer.Add(self.panel, 1, wx.EXPAND)
        self.SetSizer(frame_sizer)

        self.Fit()

    def load_files_event(self, event):
        dir_dialog = wx.DirDialog(self, "Choose a directory", style=wx.DD_DEFAULT_STYLE)
        if dir_dialog.ShowModal() == wx.ID_OK:
            self.folder_path = dir_dialog.GetPath()
            print(f"Selected folder: {self.folder_path}")
        dir_dialog.Destroy()

if __name__ == "__main__":
    app = wx.App()
    frame = MyFrame(None)
    frame.Show()
    app.MainLoop()
