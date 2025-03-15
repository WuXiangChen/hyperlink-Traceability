import wx

class MyFrame(wx.Frame):
    def __init__(self, parent):
        super().__init__(parent, title="My Frame")

        # Create the button
        self.checkAsTure = wx.Button(self, wx.ID_ANY, u"checkAsTure", wx.DefaultPosition, wx.DefaultSize, 0)

        # Bind the button click event
        self.Bind(wx.EVT_BUTTON, self.on_button_click, self.checkAsTure)

        # Create the accelerator table
        accelerator_table = wx.AcceleratorTable([
            (wx.ACCEL_NORMAL, ord('1'), self.checkAsTure.GetId())
        ])

        # Set the accelerator table
        self.SetAcceleratorTable(accelerator_table)

    def on_button_click(self, event):
        print("Button clicked!")

if __name__ == "__main__":
    app = wx.App()
    frame = MyFrame(None)
    frame.Show()
    app.MainLoop()
