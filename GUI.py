# created by omar aboulubdeh (wizard.24h@gmail.com)
import tkinter
import reGesture
import webbrowser
from ttkthemes import themed_tk as tk
from tkinter import ttk
from PIL import Image
from PIL import ImageTk


# AvailablePalmes = main.AvailablePalms()
reGesture.start()
AvailablePalmes = reGesture.available_palms()
# scale1 = None
# scale2 = None
# scale3 = None
# scale4 = None
PalmMenu = None

# sensitivity = 0
# smoothness = 0
# accuracy = 0
# threshold = 0

# def ButtonsHandler(name):
# main.Button(name)

# def ChoosePalm(name):
#     main.ChoosePalm(name)
def setShortCut(name, value):
    v = value.get()
    reGesture.set_shortcut(name,v)

def AddnewPalm():
    PalmName = PalmTextBox.get()
    reGesture.create_palm(PalmName)
    PalmMenu['menu'].delete(0, 'end')
    global AvailablePalmes
    AvailablePalmes = reGesture.available_palms()
    for p in AvailablePalmes:
        PalmMenu['menu'].add_command(label=p, command=tkinter._setit(DefaultPalm, p))
    print(AvailablePalmes)

# def GetScaleValue(value):
    # if scale_name == 'scale1':
    #     reGesture.slider_changed(scale_name,value)
    # if scale_name == 'scale2':
    #     reGesture.slider_changed(scale_name,value)
    # if scale_name == 'scale3':
    #     reGesture.slider_changed(scale_name,value)
    # if scale_name == 'scale4':
    # reGesture.slider_changed(value)
    # print(scale1.get())

def scale1_changed(value):
    reGesture.slider_changed("sensitivity",value)
def scale2_changed(value):
    reGesture.slider_changed("smoothness",value)
def scale3_changed(value):
    reGesture.slider_changed("accuracy",value)
def scale4_changed(value):
    reGesture.slider_changed("thresshold",value)
def GetOptionsValue(value):
    reGesture.choose_palm(value)

def ToggleChannels():
    ViewChannels.config('text')
    if ViewChannels('text')[-1] == 'View Channels':
        ViewChannels.config(text='Close Channels')
        reGesture.toggle_channels = ('true')
    else:
        ViewChannels.config(text='View Channels')
        reGesture.toggle_channels = ('false')
    if ViewChannels('text')[-1] == 'Close Channels':
        ViewChannels.config(text='View Channels')
        reGesture.toggle_channels = ('false')
    else:
        ViewChannels.config(text='Close Channels')
        reGesture.toggle_channels = ('true')


window = tk.ThemedTk()
window.get_themes()
window.set_theme("black")
style = ttk.Style()
window.geometry("910x480")
window.title("Senior Project")
rows = 0
while rows < 50:
    window.rowconfigure(rows, weight=1)
    window.columnconfigure(rows, weight=1)
    rows += 1
nb = ttk.Notebook(window)
nb.grid(row=1, column=0, columnspan=50, rowspan=49, sticky="NESW")
page1 = ttk.Frame(nb)
nb.add(page1, text="Shortcuts")
page2 = ttk.Frame(nb)
nb.add(page2, text="Settings")
page3 = ttk.Frame(nb)
nb.add(page3, text="About us")
page4 = ttk.Frame(nb)
nb.add(page4, text="Palm Settings")
backgroundimage1 = Image.open("./Images/gradiant_background.jpg")
backgroundImage = ImageTk.PhotoImage(backgroundimage1)
# window.configure(background="black")
C = tkinter.Label(page4, image=backgroundImage)
C.place(relx=.5, rely=.5, anchor="center")
C.lower()
C.rowconfigure(0, weight=100)
C.rowconfigure(3, weight=100)
C.columnconfigure(0, weight=100)
C.columnconfigure(3, weight=100)
C = tkinter.Label(page2, image=backgroundImage)
C.place(relx=.5, rely=.5, anchor="center")
C.lower()
C.rowconfigure(0, weight=100)
C.rowconfigure(3, weight=100)
C.columnconfigure(0, weight=100)
C.columnconfigure(3, weight=100)
C = tkinter.Label(page3, image=backgroundImage)
C.place(relx=.5, rely=.5, anchor="center")
C.lower()
C.rowconfigure(0, weight=100)
C.rowconfigure(3, weight=100)
C.columnconfigure(0, weight=100)
C.columnconfigure(3, weight=100)
C = tkinter.Label(page1, image=backgroundImage)
C.place(relx=.5, rely=.5, anchor="center")
C.lower()
C.rowconfigure(0, weight=100)
C.rowconfigure(3, weight=100)
C.columnconfigure(0, weight=100)
C.columnconfigure(3, weight=100)
C = tkinter.Label(page4, image=backgroundImage)

img1 = Image.open("./Images/two.png")
img1 = img1.resize((100, 100))
new_img1 = ImageTk.PhotoImage(img1)
img2 = Image.open("./Images/three.jpg").convert("RGBA")
img2 = img2.resize((100, 100), Image.ANTIALIAS)
new_img2 = ImageTk.PhotoImage(img2)
img3 = Image.open("./Images/four.png").convert("RGBA")
img3 = img3.resize((100, 100), Image.ANTIALIAS)
new_img3 = ImageTk.PhotoImage(img3)
img4 = Image.open("./Images/phone.png").convert("RGBA")
img4 = img4.resize((100, 100), Image.ANTIALIAS)
new_img4 = ImageTk.PhotoImage(img4)

AIU_logo = Image.open("./Images/220px-Arab_International_University_logo.png")
AIU_logo = AIU_logo.resize((200, 200), Image.ANTIALIAS)
new_AIU_logo = ImageTk.PhotoImage(AIU_logo)
# label1 = tkinter.Label(window,text = "Hand Gesturing",font = ("arial",16,"bold")).grid(row=0,column=22)
Label2 = tkinter.Label(page1, text="Hand1", image=new_img1, bg="#282c31", font=("arial", 16, "bold"), borderwidth=0)
Label3 = tkinter.Label(page1, text="Hand2", image=new_img2, bg="#212529", font=("arial", 16, "bold"))
Label4 = tkinter.Label(page1, text="Hand2", image=new_img3, fg="white", bg="#212529", font=("arial", 16, "bold"))
Label5 = tkinter.Label(page1, text="Hand4", image=new_img4, bg="#212529")
AIULOGOLABEL = tkinter.Label(page3, image=new_AIU_logo, text="Aiulolo", bg="#1d2024")

Label2.grid(row=10)
Label3.grid(row=10, column=2)
Label4.grid(row=13)
Label5.grid(row=13,column=2)

AIULOGOLABEL.place(relx=.4, rely=.5)
hand1 = tkinter.StringVar()
hand2 = tkinter.StringVar()
hand3 = tkinter.StringVar()
hand4 = tkinter.StringVar()


hand1.trace("w", lambda name, index, mode, hand1=hand1: setShortCut('hand1', hand1))
hand2.trace("w", lambda name, index, mode, hand2=hand2: setShortCut('hand2', hand2))
hand3.trace("w", lambda name, index, mode, hand3=hand3: setShortCut('hand3', hand3))
hand4.trace("w", lambda name, index, mode, hand4=hand4: setShortCut('hand4', hand4))

e1 = tkinter.Entry(page1,textvariable=hand1)
e2 = tkinter.Entry(page1,textvariable=hand2)
e3 = tkinter.Entry(page1,textvariable=hand3)
e4 = tkinter.Entry(page1,textvariable=hand4)
PalmTextBox = tkinter.Entry(page4)
e1.grid(row=11,)
e2.grid(row=11, column=2)
e3.grid(row=14)
e4.grid(row=14, column=2)
PalmTextBox.place(relx=0.2, rely=.25)
StartButton = ttk.Button(page1, text="Start", width=5, command=lambda :reGesture.button_clicked("start")).grid(
    row=16,
    column=1,
    ipady=15,
    ipadx=5)
# QuitButton = tkinter.Button(page1,text="Quit",height=2,width=5,command=window.destroy,font="bold",borderwidth="4").grid(row=35,column=0)
QuitButton2 = ttk.Button(page1, text="Quit", width=5, command=window.destroy).grid(
    row=16,
    column=0,
    ipady=15,
    ipadx=5)
e1.insert(10, "start www.facebook.com")
e2.insert(10, "start www.google.com")
e3.insert(10, "start www.youtube.com")
e4.insert(10, "start www.twitter.com")
PalmTextBox.insert(10, 'Orange')
scale1 = tkinter.Scale(page2, from_=1, to=20, orient=tkinter.HORIZONTAL, bg="#282c31", length=200,
                       label="Mouse Sensitivity", fg="white", command=scale1_changed).grid(row=2, column=0)
# print(scale1.get())
scale2 = tkinter.Scale(page2, from_=1, to=5, orient=tkinter.HORIZONTAL, bg="#212529", length=200, label="Smoothnes",
                       fg="white", command=scale2_changed).grid(row=10, column=0)
scale3 = tkinter.Scale(page2, from_=1, to=10, orient=tkinter.HORIZONTAL, bg="#1f2226", length=200, label="Accuracy",
                       fg="white", command=scale3_changed).grid(row=18, column=0)
scale4 = tkinter.Scale(page2, from_=0, to=5, orient=tkinter.HORIZONTAL, bg="#1f2226", length=200,
                       label="Button Threshold", fg="white", command=scale4_changed).grid(row=19, column=0)
# ApplyButton1 = ttk.Button(page2,text="Apply",command=apply)
LeftorRight = tkinter.StringVar()
LeftorRight.set("left")
LeftRadioButton1 = tkinter.Radiobutton(page2, text="Left Hand", variable=LeftorRight, value="left", bg="#1f2226",
                                       fg="white", command=lambda : reGesture.button_clicked("left_hand")).place(relx=.0, rely=.6)
LeftRadioButton2 = tkinter.Radiobutton(page2, text="Right Hand", variable=LeftorRight, value="right", bg="#1f2226",
                                       fg="white", command=lambda : reGesture.button_clicked("right_hand")).place(relx=.1, rely=.6)
ViewChannels = ttk.Button(page2, text="View Channels", command=lambda : reGesture.button_clicked("view_channels")).place(relx=.0,
                                                                                               rely=.55)  # grid(row=26,column=0)
NewPalmButton = ttk.Button(page4, text="New Palm", command=lambda: AddnewPalm()).place(relx=0.1,
                                                                                       rely=.25)  # grid(row=26,column=4)
Aboud = tkinter.Label(page3, text="Abdallah Madani", font=("arial", 15, "bold"), fg="white", bg="#282c31").place(
    relx=0.43, rely=0.1)
Aman = tkinter.Label(page3, text="Aman AlSafadi", font=("arial", 15, "bold"), fg="white", bg="#282c31").place(
    relx=0.43, rely=0.2)
Alaa = tkinter.Label(page3, text="Alaa Alhafez", font=("arial", 15, "bold"), fg="white", bg="#282c31").place(
    relx=0.43, rely=0.3)

Omar = tkinter.Label(page3, text="Omar Aboulabdeh",cursor='hand2', font=("arial", 15, "bold"), fg="white", bg="#282c31")
Omar.bind("<Button-1>", lambda e: webbrowser.open_new('https://www.facebook.com/omar.aboulabdeh'))
Omar.place(
    relx=0.43, rely=0.4)
DefaultPalm = tkinter.StringVar()
DefaultPalm.set(AvailablePalmes[0])
PalmMenu = ttk.OptionMenu(page4, DefaultPalm, *AvailablePalmes, command=GetOptionsValue)
tkinter.Label(page4, text="Choose Palm", fg="white", bg="#282c31").place(relx=0.05, rely=0.35)
PalmMenu.place(relx=0.15, rely=0.35)
window.mainloop()