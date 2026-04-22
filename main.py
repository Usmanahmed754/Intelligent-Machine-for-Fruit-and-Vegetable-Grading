from tkinter import*
from tkinter import messagebox, ttk
import mysql.connector
import re
import serial
import cv2
import cv2 as cv
import torch
import numpy as np
import time
from sqlalchemy import create_engine
from datetime import datetime
import argparse
from typing import Tuple
from utils.edge import *
from utils.general import *
from utils.threshold import *

# DEFINE THE DATABASE CREDENTIALS
user1 = 'root'
password1 = 'admin123'
host1 = '127.0.0.1'
port1 = 3306
database1 = 'mainsys'



 
# PYTHON FUNCTION TO CONNECT TO THE MYSQL DATABASE AND
# RETURN THE SQLACHEMY ENGINE OBJECT
def get_connection():
    return create_engine(
        url="mysql+pymysql://{0}:{1}@{2}:{3}/{4}".format(
            user1, password1, host1, port1, database1
        )
    )

mydb = get_connection()

#path = 'best.pt'
#model = torch.hub.load('ultralytics/yolov5', 'custom', path, force_reload=True)

Conf_threshold = 0.4
NMS_threshold = 0.4
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
          (255, 255, 0), (255, 0, 255), (0, 255, 255)]

class_name = []
with open('classes.txt', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]
# print(class_name)
net = cv.dnn.readNet('yolov4.weights', 'yolov4.cfg')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)


regex = '^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}$'

ser = serial.Serial('COM4', 9600)

cap=cv2.VideoCapture(1)
# Windows Size and Placement
main = Tk()
main.grid_rowconfigure(0, weight=1)
main.grid_columnconfigure(0, weight=1)
height = 650
width = 1240
x = (main.winfo_screenwidth()//2) - (width//2)
y = (main.winfo_screenheight()//4) - (height//4)
main.geometry('{}x{}+{}+{}'.format(width, height, x, y))

main.title('Intelligent Machine for Fruit & Vegetable Grading')

# Navigating through windows
sign_in = Frame(main)
sign_up = Frame(main)
dashboard = Frame(main)
manual = Frame(main)

categoryVar= IntVar()
classificationVar= IntVar()

for frame in (sign_in, sign_up, dashboard, manual):
	frame.grid(row=0, column=0, sticky='nsew')

def show_frame(frame):
	frame.tkraise();


def logUdate(label):
    # establish connection 
    database = mysql.connector.connect( 
        host=host1, 
        user=user1, 
        password=password1, 
        database=database1
    ) 
      
    # creating cursor object 
    cur_object = database.cursor() 
    find = "SELECT * FROM data_logs WHERE fruitOrVegetable LIKE %s limit 1" 
    cur_object.execute(find, ("%" + label + "%",)) 
    data = cur_object.fetchall()

    if data : 
        for i in data:
            count=int(i[2])
            count = count + 1
            now = datetime.now()
            dateString = now.strftime('%Y-%m-%d %H:%M:%S')
            sql = """ UPDATE data_logs SET counts = %s, lastUpdate = %s WHERE idlogs = %s """
            data = (count, dateString, i[0])
            cur_object.execute(sql, data)
            affected_rows = cur_object.rowcount
            database.commit()
            print("Update Sucessful")
    else :
        print("Not Found")
        now = datetime.now()
        dateString = now.strftime('%Y-%m-%d %H:%M:%S')
        sql = "INSERT INTO data_logs (fruitOrVegetable, counts, lastUpdate) VALUES (%s, %s, %s)"
        val = (label, "1", dateString)
        cur_object.execute(sql, val)
        database.commit()
        print("New data added")    


def open_logs():
    my_w= Toplevel(main)
    my_w.title("ActivityLogs")  
    # Using treeview widget
    trv = ttk.Treeview(my_w, selectmode ='browse')
    trv.grid(row=1,column=1,padx=40,pady=40)

    # number of columns
    trv["columns"] = ("1", "2", "3", "4")

    # Defining heading
    trv['show'] = 'headings'

    # width of columns and alignment 
    trv.column("1", width = 30, anchor ='c')
    trv.column("2", width = 180, anchor ='c')
    trv.column("3", width = 80, anchor ='c')
    trv.column("4", width = 180, anchor ='c')
    # Headings  
    # respective columns
    trv.heading("1", text ="id")
    trv.heading("2", text ="Fruits/Vegetable Names")
    trv.heading("3", text ="Counts")
    trv.heading("4", text ="Last Update")
    # getting data from MySQL student table 
    r_set=mydb.execute('''SELECT * from data_logs LIMIT 0,10''')
    for dt in r_set: 
        trv.insert("", 'end',iid=dt[0], text=dt[0],
                   values =(dt[0],dt[1],dt[2],dt[3]))


show_frame(sign_in)


# ====================================================
# ================== SIGN UP PAGE ===================
# ====================================================

def email_validate(u_input):
    if(not re.search(regex,u_input) and not u_input.isalpha):
        messagebox.showerror("Error", "Please enter valid Email")
my_valid = sign_up.register(email_validate)

# Sign Up Text Variable
FirstName = StringVar()
LastName = StringVar()
Email = StringVar()
Password = StringVar()
ConfirmPassword = StringVar()

sign_up.configure(bg="#525561")

# ================Background Image ====================
backgroundImage = PhotoImage(file="assets\\image_1.png")
bg_image = Label(
    sign_up,
    image=backgroundImage,
    bg="#525561"
)
bg_image.place(x=120, y=28)

# ================ Header Text Left ====================
headerText_image_left = PhotoImage(file="assets\\headerText_image.png")
headerText_image_label1 = Label(
    bg_image,
    image=headerText_image_left,
    bg="#272A37"
)
headerText_image_label1.place(x=60, y=45)

headerText1 = Label(
    bg_image,
    text="FV Grading Sys",
    fg="#FFFFFF",
    font=("yu gothic ui bold", 20 * -1),
    bg="#272A37"
)
headerText1.place(x=110, y=45)

# ================ Header Text Right ====================
headerText_image_right = PhotoImage(file="assets\\headerText_image.png")
headerText_image_label2 = Label(
    bg_image,
    image=headerText_image_right,
    bg="#272A37"
)
headerText_image_label2.place(x=400, y=45)

headerText2 = Label(
    bg_image,
    anchor="nw",
    text="Enter credentials to access the system.",
    fg="#FFFFFF",
    font=("yu gothic ui Bold", 20 * -1),
    bg="#272A37"
)
headerText2.place(x=450, y=45)

# ================ CREATE ACCOUNT HEADER ====================
createAccount_header = Label(
    bg_image,
    text="Create new account",
    fg="#FFFFFF",
    font=("yu gothic ui Bold", 28 * -1),
    bg="#272A37"
)
createAccount_header.place(x=75, y=121)

# ================ ALREADY HAVE AN ACCOUNT TEXT ====================
text = Label(
    bg_image,
    text="Already a member?",
    fg="#FFFFFF",
    font=("yu gothic ui Regular", 15 * -1),
    bg="#272A37"
)
text.place(x=75, y=187)

# ================ GO TO LOGIN ====================
switchLogin = Button(
    bg_image,
    text="Login",
    fg="#206DB4",
    font=("yu gothic ui Bold", 15 * -1),
    bg="#272A37",
    bd=0,
    cursor="hand2",
    activebackground="#272A37",
    activeforeground="#ffffff",
    command=lambda : show_frame(sign_in)
)
switchLogin.place(x=230, y=185, width=50, height=35)

# ================ First Name Section ====================
firstName_image = PhotoImage(file="assets\\input_img.png")
firstName_image_Label = Label(
    bg_image,
    image=firstName_image,
    bg="#272A37"
)
firstName_image_Label.place(x=80, y=242)

firstName_text = Label(
    firstName_image_Label,
    text="First name",
    fg="#FFFFFF",
    font=("yu gothic ui SemiBold", 13 * -1),
    bg="#3D404B"
)
firstName_text.place(x=25, y=0)

firstName_icon = PhotoImage(file="assets\\name_icon.png")
firstName_icon_Label = Label(
    firstName_image_Label,
    image=firstName_icon,
    bg="#3D404B"
)
firstName_icon_Label.place(x=159, y=15)

firstName_entry = Entry(
    firstName_image_Label,
    bd="0",
    bg="#3D404B",
    highlightthickness=0,
    fg="white",
    textvariable=FirstName,
    font=("yu gothic ui SemiBold", 16 * -1),
)
firstName_entry.place(x=8, y=17, width=140, height=27)


# ================ Last Name Section ====================
lastName_image = PhotoImage(file="assets\\input_img.png")
lastName_image_Label = Label(
    bg_image,
    image=lastName_image,
    bg="#272A37"
)
lastName_image_Label.place(x=293, y=242)

lastName_text = Label(
    lastName_image_Label,
    text="Last name",
    fg="#FFFFFF",
    font=("yu gothic ui SemiBold", 13 * -1),
    bg="#3D404B"
)
lastName_text.place(x=25, y=0)

lastName_icon = PhotoImage(file="assets\\name_icon.png")
lastName_icon_Label = Label(
    lastName_image_Label,
    image=lastName_icon,
    bg="#3D404B"
)
lastName_icon_Label.place(x=159, y=15)

lastName_entry = Entry(
    lastName_image_Label,
    bd=0,
    bg="#3D404B",
    highlightthickness=0,
    fg="white",
    textvariable=LastName,
    font=("yu gothic ui SemiBold", 16 * -1),
)
lastName_entry.place(x=8, y=17, width=140, height=27)

# ================ Email Name Section ====================
emailName_image = PhotoImage(file="assets\\email.png")
emailName_image_Label = Label(
    bg_image,
    image=emailName_image,
    bg="#272A37"
)
emailName_image_Label.place(x=80, y=311)

emailName_text = Label(
    emailName_image_Label,
    text="Email account",
    fg="#FFFFFF",
    font=("yu gothic ui SemiBold", 13 * -1),
    bg="#3D404B"
)
emailName_text.place(x=25, y=0)

emailName_icon = PhotoImage(file="assets\\email-icon.png")
emailName_icon_Label = Label(
    emailName_image_Label,
    image=emailName_icon,
    bg="#3D404B"
)
emailName_icon_Label.place(x=370, y=15)

emailName_entry = Entry(
    emailName_image_Label,
    bd=0,
    bg="#3D404B",
    highlightthickness=0,
    fg="white",
    textvariable=Email,
    font=("yu gothic ui SemiBold", 16 * -1),
    validate='focusout',
    validatecommand=(my_valid, '%P')
)
emailName_entry.place(x=8, y=17, width=354, height=27)


# ================ Password Name Section ====================
passwordName_image = PhotoImage(file="assets\\input_img.png")
passwordName_image_Label = Label(
    bg_image,
    image=passwordName_image,
    bg="#272A37"
)
passwordName_image_Label.place(x=80, y=380)

passwordName_text = Label(
    passwordName_image_Label,
    text="Password",
    fg="#FFFFFF",
    font=("yu gothic ui SemiBold", 13 * -1),
    bg="#3D404B"
)
passwordName_text.place(x=25, y=0)

passwordName_icon = PhotoImage(file="assets\\pass-icon.png")
passwordName_icon_Label = Label(
    passwordName_image_Label,
    image=passwordName_icon,
    bg="#3D404B"
)
passwordName_icon_Label.place(x=159, y=15)

passwordName_entry = Entry(
    passwordName_image_Label,
    bd=0,
    bg="#3D404B",
    highlightthickness=0,
    fg="white",
    textvariable=Password,
    show='*',
    font=("yu gothic ui SemiBold", 16 * -1),
)
passwordName_entry.place(x=8, y=17, width=140, height=27)


# ================ Confirm Password Name Section ====================
confirm_passwordName_image = PhotoImage(file="assets\\input_img.png")
confirm_passwordName_image_Label = Label(
    bg_image,
    image=confirm_passwordName_image,
    bg="#272A37"
)
confirm_passwordName_image_Label.place(x=293, y=380)

confirm_passwordName_text = Label(
    confirm_passwordName_image_Label,
    text="Confirm Password",
    fg="#FFFFFF",
    font=("yu gothic ui SemiBold", 13 * -1),
    bg="#3D404B"
)
confirm_passwordName_text.place(x=25, y=0)

confirm_passwordName_icon = PhotoImage(file="assets\\pass-icon.png")
confirm_passwordName_icon_Label = Label(
    confirm_passwordName_image_Label,
    image=confirm_passwordName_icon,
    bg="#3D404B"
)
confirm_passwordName_icon_Label.place(x=159, y=15)

confirm_passwordName_entry = Entry(
    confirm_passwordName_image_Label,
    bd=0,
    bg="#3D404B",
    highlightthickness=0,
    fg="white",
    textvariable=ConfirmPassword,
    show='*',
    font=("yu gothic ui SemiBold", 16 * -1),
)
confirm_passwordName_entry.place(x=8, y=17, width=140, height=27)

# =============== Submit Button ====================
submit_buttonImage = PhotoImage(
    file="assets\\button_1.png")
submit_button = Button(
    bg_image,
    image=submit_buttonImage,
    borderwidth=0,
    highlightthickness=0,
    relief="flat",
    activebackground="#272A37",
    cursor="hand2",
    command=lambda: signup()
)
submit_button .place(x=130, y=460, width=333, height=65)

# ================ Header Text Down ====================
headerText_image_down = PhotoImage(file="assets\\headerText_image.png")
headerText_image_label3 = Label(
    bg_image,
    image=headerText_image_down,
    bg="#272A37"
)
headerText_image_label3.place(x=650, y=530)

headerText3 = Label(
    bg_image,
    text="Powered by Farjad",
    fg="#FFFFFF",
    font=("yu gothic ui bold", 20 * -1),
    bg="#272A37"
)
headerText3.place(x=700, y=530)


# clear sign up fields
def clear():
    FirstName.set('')
    LastName.set('')
    Email.set('')
    Password.set('')
    ConfirmPassword.set('')

# ====================================================
# ======= DATABASE CONNECTION FOR SIGN UP ===========
# ====================================================
def signup():
    if firstName_entry.get() == "" or lastName_entry.get() == "" or emailName_entry.get() == "" or passwordName_entry.\
        get() == "" or confirm_passwordName_entry.get() == "":
        messagebox.showerror("Error", "All Fields are Required")

    elif passwordName_entry.get() != confirm_passwordName_entry.get():
        messagebox.showerror("Error", "Password Didn't Match")

    elif not firstName_entry.get().isalpha() or not lastName_entry.get().isalpha():
        messagebox.showerror("Error", "Only letters are allowed in First Name or Last Name")

    elif not re.fullmatch(r'[A-Za-z0-9@#$%^&+=]{8,}', passwordName_entry.get()):
        messagebox.showerror("Error", "Your password must be strong with special characters.")

    else:
        if re.fullmatch(regex, emailName_entry.get()):
            try:
                connection = mysql.connector.connect(host=host1, user=user1, password=password1, database=database1)
                cur = connection.cursor()
                sql = "INSERT INTO users(FirstName, LastName, EMAIL, Password) VALUES(%s, %s, %s, %s)"
                val = (firstName_entry.get(), lastName_entry.get(), emailName_entry.get(), passwordName_entry.get())
                cur.execute(sql, val)
                connection.commit()
                connection.close()
                clear()
                messagebox.showinfo("Success", "New Account Created Successfully")

            except Exception as ex:
                messagebox.showerror("Error", "Something went wrong try again")

        else:
            messagebox.showerror("Error", "Please enter valid email.")


# ====================================================
# ================== SIGN IN PAGE ===================
# ====================================================

# login text variables
email = StringVar()
password = StringVar()


sign_in.configure(bg="#525561")

# ================Background Image ====================
Login_backgroundImage = PhotoImage(file="assets\\image_1.png")
bg_imageLogin = Label(
    sign_in,
    image=Login_backgroundImage,
    bg="#525561"
)
bg_imageLogin.place(x=120, y=28)

# ================ Header Text Left ====================
Login_headerText_image_left = PhotoImage(file="assets\\headerText_image.png")
Login_headerText_image_label1 = Label(
    bg_imageLogin,
    image=Login_headerText_image_left,
    bg="#272A37"
)
Login_headerText_image_label1.place(x=60, y=45)

Login_headerText1 = Label(
    bg_imageLogin,
    text="FV Grading Sys",
    fg="#FFFFFF",
    font=("yu gothic ui bold", 20 * -1),
    bg="#272A37"
)
Login_headerText1.place(x=110, y=45)

# ================ Header Text Right ====================
Login_headerText_image_right = PhotoImage(file="assets\\headerText_image.png")
Login_headerText_image_label2 = Label(
    bg_imageLogin,
    image=Login_headerText_image_right,
    bg="#272A37"
)
Login_headerText_image_label2.place(x=400, y=45)

Login_headerText2 = Label(
    bg_imageLogin,
    anchor="nw",
    text="Enter credentials to access the system",
    fg="#FFFFFF",
    font=("yu gothic ui Bold", 20 * -1),
    bg="#272A37"
)
Login_headerText2.place(x=450, y=45)

# ================ LOGIN TO ACCOUNT HEADER ====================
loginAccount_header = Label(
    bg_imageLogin,
    text="Login to continue",
    fg="#FFFFFF",
    font=("yu gothic ui Bold", 28 * -1),
    bg="#272A37"
)
loginAccount_header.place(x=75, y=121)

# ================ NOT A MEMBER TEXT ====================
loginText = Label(
    bg_imageLogin,
    text="Not a member?",
    fg="#FFFFFF",
    font=("yu gothic ui Regular", 15 * -1),
    bg="#272A37"
)
loginText.place(x=75, y=187)

# ================ GO TO SIGN UP ====================
switchSignup = Button(
    bg_imageLogin,
    text="Sign Up",
    fg="#206DB4",
    font=("yu gothic ui Bold", 15 * -1),
    bg="#272A37",
    bd=0,
    cursor="hand2",
    activebackground="#272A37",
    activeforeground="#ffffff",
    command=lambda : show_frame(sign_up)
)
switchSignup.place(x=220, y=185, width=70, height=35)


# ================ Email Name Section ====================
Login_emailName_image = PhotoImage(file="assets\\email.png")
Login_emailName_image_Label = Label(
    bg_imageLogin,
    image=Login_emailName_image,
    bg="#272A37"
)
Login_emailName_image_Label.place(x=76, y=242)

Login_emailName_text = Label(
    Login_emailName_image_Label,
    text="Email account",
    fg="#FFFFFF",
    font=("yu gothic ui SemiBold", 13 * -1),
    bg="#3D404B"
)
Login_emailName_text.place(x=25, y=0)

Login_emailName_icon = PhotoImage(file="assets\\email-icon.png")
Login_emailName_icon_Label = Label(
    Login_emailName_image_Label,
    image=Login_emailName_icon,
    bg="#3D404B"
)
Login_emailName_icon_Label.place(x=370, y=15)

Login_emailName_entry = Entry(
    Login_emailName_image_Label,
    bd=0,
    bg="#3D404B",
    highlightthickness=0,
    fg="white",
    textvariable=email,
    font=("yu gothic ui SemiBold", 16 * -1),
)
Login_emailName_entry.place(x=8, y=17, width=354, height=27)


# ================ Password Name Section ====================
Login_passwordName_image = PhotoImage(file="assets\\email.png")
Login_passwordName_image_Label = Label(
    bg_imageLogin,
    image=Login_passwordName_image,
    bg="#272A37"
)
Login_passwordName_image_Label.place(x=80, y=330)

Login_passwordName_text = Label(
    Login_passwordName_image_Label,
    text="Password",
    fg="#FFFFFF",
    font=("yu gothic ui SemiBold", 13 * -1),
    bg="#3D404B"
)
Login_passwordName_text.place(x=25, y=0)

Login_passwordName_icon = PhotoImage(file="assets\\pass-icon.png")
Login_passwordName_icon_Label = Label(
    Login_passwordName_image_Label,
    image=Login_passwordName_icon,
    bg="#3D404B"
)
Login_passwordName_icon_Label.place(x=370, y=15)

Login_passwordName_entry = Entry(
    Login_passwordName_image_Label,
    bd=0,
    bg="#3D404B",
    highlightthickness=0,
    fg="white",
    textvariable=password,
    show='*',
    font=("yu gothic ui SemiBold", 16 * -1),
)
Login_passwordName_entry.place(x=8, y=17, width=354, height=27)

# =============== Submit Button ====================
Login_button_image_1 = PhotoImage(
    file="assets\\button_1.png")
Login_button_1 = Button(
    bg_imageLogin,
    image=Login_button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: login(),
    relief="flat",
    activebackground="#272A37",
    cursor="hand2",
)
Login_button_1.place(x=120, y=445, width=333, height=65)

# ================ Header Text Down ====================
Login_headerText_image_down = PhotoImage(file="assets\\headerText_image.png")
Login_headerText_image_label3 = Label(
    bg_imageLogin,
    image=Login_headerText_image_down,
    bg="#272A37"
)
Login_headerText_image_label3.place(x=650, y=530)

Login_headerText3 = Label(
    bg_imageLogin,
    text="Powered by Farjad",
    fg="#FFFFFF",
    font=("yu gothic ui bold", 20 * -1),
    bg="#272A37"
)
Login_headerText3.place(x=700, y=530)

def clear_login():
    email.set('')
    password.set('')

def login():
    connection = mysql.connector.connect(host=host1, user=user1, password=password1, database=database1)
    if Login_emailName_entry.get() == "" or Login_passwordName_entry.get() == "":
        messagebox.showerror("Error", "All Fields are Required")
    else:
        cur = connection.cursor()
        sql_query = "SELECT *FROM users WHERE EMAIL ='%s' AND password ='%s'" % (Login_emailName_entry.get(), Login_passwordName_entry.get())
        cur.execute(sql_query)

        result = cur.fetchall()

        if result:
            messagebox.showinfo("Success", "Logged in Successfully")
            clear_login()
            show_frame(dashboard)
        else:
            messagebox.showerror("Error", "Sorry, User not found")
    connection.commit()
    connection.close()



# ================ Forgot Password ====================
forgotPassword = Button(
    bg_imageLogin,
    text="Forgot Password",
    fg="#206DB4",
    font=("yu gothic ui Bold", 15 * -1),
    bg="#272A37",
    bd=0,
    activebackground="#272A37",
    activeforeground="#ffffff",
    cursor="hand2",
    command=lambda: forgot_password(),
)
forgotPassword.place(x=210, y=400, width=150, height=35)


def forgot_password():

    win = Toplevel()
    window_width = 350
    window_height = 350
    screen_width = win.winfo_screenwidth()
    screen_height = win.winfo_screenheight()
    position_top = int(screen_height / 4 - window_height / 4)
    position_right = int(screen_width / 2 - window_width / 2)
    win.geometry(f'{window_width}x{window_height}+{position_right}+{position_top}')

    win.title('Forgot Password')
    # win.iconbitmap('images\\aa.ico')
    win.configure(background='#272A37')
    win.resizable(False, False)

    # ====== Email ====================
    email_entry3 = Entry(win, bg="#3D404B", font=("yu gothic ui semibold", 12), highlightthickness=1,
                         bd=0)
    email_entry3.place(x=40, y=80, width=256, height=50)
    email_entry3.config(highlightbackground="#3D404B", highlightcolor="#206DB4")
    email_label3 = Label(win, text='• Email', fg="#FFFFFF", bg='#272A37',
                         font=("yu gothic ui", 11, 'bold'))
    email_label3.place(x=40, y=50)

    # ====  New Password ==================
    new_password_entry = Entry(win, bg="#3D404B", font=("yu gothic ui semibold", 12), show='•', highlightthickness=1,
                               bd=0)
    new_password_entry.place(x=40, y=180, width=256, height=50)
    new_password_entry.config(highlightbackground="#3D404B", highlightcolor="#206DB4")
    new_password_label = Label(win, text='• New Password', fg="#FFFFFF", bg='#272A37',
                               font=("yu gothic ui", 11, 'bold'))
    new_password_label.place(x=40, y=150)

    # ======= Update password Button ============
    update_pass = Button(win, fg='#f8f8f8', text='Update Password', bg='#1D90F5', font=("yu gothic ui", 12, "bold"),
                         cursor='hand2', relief="flat", bd=0, highlightthickness=0, activebackground="#1D90F5")
    update_pass.place(x=40, y=260, width=256, height=45)





# classification
def detect_defects(colour_image: np.ndarray, nir_image: np.ndarray, image_name: str = '', tweak_factor: float = .3,
                   sigma: float = 1., threshold_1: int = 60, threshold_2: int = 130,
                   verbose: bool = True) -> Tuple[int, np.ndarray, np.ndarray]:

    # Filter the NIR image by median blur
    f_nir_image = cv.medianBlur(nir_image, 5)

    # Get the fruit mask through Tweaked Otsu's algorithm
    mask = get_fruit_segmentation_mask(f_nir_image, ThresholdingMethod.TWEAKED_OTSU, tweak_factor=tweak_factor)

    # Apply the mask to the filtered NIR image
    m_nir_image = apply_mask_to_image(f_nir_image, mask)

    # Get the edge mask through Gaussian Blur and Canny's method
    edge_mask = apply_gaussian_blur_and_canny(m_nir_image, sigma, threshold_1, threshold_2)

    # Erode the mask to get rid of the edges of the bound of the fruit
    erode_element = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
    eroded_mask = cv.erode(mask, erode_element)

    # Remove background edges from the edge mask
    edge_mask = apply_mask_to_image(edge_mask, eroded_mask)

    # Apply Closing operation to fill the defects according to the edges and obtain the defect mask
    close_element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (50, 50))
    defect_mask = cv.morphologyEx(edge_mask, cv.MORPH_CLOSE, close_element)
    defect_mask = cv.medianBlur(defect_mask, 7)

    # Perform a connected components labeling to detect the defects
    retval, labels, stats, centroids = cv.connectedComponentsWithStats(defect_mask)

    print(f'Detected {retval - 1} defect{"" if retval - 1 == 1 else "s"} for image {image_name}.')

    if verbose:
        

        # Get highlighted defects on the fruit
        highlighted_roi = get_highlighted_roi_by_mask(colour_image, defect_mask, 'red')

        circled_defects = np.copy(colour_image)

        for i in range(1, retval):
            s = stats[i]
            # Draw a red ellipse around the defect
            cv.ellipse(circled_defects, center=tuple(int(c) for c in centroids[i]),
                       axes=(s[cv.CC_STAT_WIDTH] // 2 + 10, s[cv.CC_STAT_HEIGHT] // 2 + 10),
                       angle=0, startAngle=0, endAngle=360, color=(0, 0, 255), thickness=3)

        plot_image_grid([highlighted_roi, circled_defects],
                        ['Detected defects ROI', 'Detected defects areas'],
                        f'Defects of the fruit {image_name}')
        
        
    return retval - 1


def classification(className, coloImage, nirImage, verbose):
        # Read colour image
    fruit_image_path = coloImage
    colour_image = cv.imread(fruit_image_path)

    # Read NIR image
    fruit_nir_image_path = nirImage
    nir_image = cv.imread(fruit_nir_image_path, cv.IMREAD_GRAYSCALE)

    tweak_factor = 0.3
    sigma = 1.0
    threshold_1 = 60
    threshold_2 = 120

    detects = detect_defects(colour_image, nir_image, image_name=className, tweak_factor=tweak_factor, sigma=sigma,
                   threshold_1=threshold_1, threshold_2=threshold_2, verbose=verbose)
    
    return detects


# ====================================================
# ================== DASHBOARD ===================
# ====================================================
def open_camera():
    guiShow = False
    detectCount = 0
    appleCount = 0
    eggplantCount = 0
    tomatoCount = 0
    timer = 0
    ser.write(bytes('S', 'UTF-8'))
    timerOn = False
    finalLabel  = 'None'
    starting_time = time.time()
    frame_counter = 0    
    while True:
        if timerOn:
            timer = timer + 1
            print(timer)

        ret, frame = cap.read()
        frame_counter += 1
        if ret == False:
            break
        classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)
        for (classid, score, box) in zip(classes, scores, boxes):
            color = COLORS[int(classid) % len(COLORS)]
            labelName = class_name[classid]

            if labelName == "apple" or labelName == "banana" or labelName == "donut" or labelName == "sports ball" or labelName == "carrot" or labelName == "mouse" :
                if labelName == "sports ball" :
                    labelName = "tomato"
                if labelName == "donut" or labelName == "mouse":
                    labelName = "eggplant"

                if classificationVar.get()==1 and timerOn == False:
                    if labelName == "banana":
                      detectCount = classification(labelName, "images/bananaDefect.png", "images/bananaDefect1.png", guiShow)
                    elif labelName == "carrot" :
                      detectCount = classification(labelName, "images/carrotUnDefect.png", "images/carrotUnDefect1.png", guiShow)
                    elif labelName == "apple" :
                        if appleCount == 0:
                           detectCount = classification(labelName, "images/appleUnDefect.png", "images/appleUnDefect1.png", guiShow)
                        else:
                           detectCount = classification(labelName, "images/appleDefect.png", "images/appleDefect1.png", guiShow)
                        appleCount = appleCount + 1
                        if(appleCount > 1 ) : 
                            appleCount = 0
                      
                    elif labelName == "eggplant" :
                        if eggplantCount == 0:
                           detectCount = classification(labelName, "images/eggplantUnDefect.png", "images/eggplantUnDefect1.png", guiShow)
                        else:
                           detectCount = classification(labelName, "images/eggplantDefect.png", "images/eggplantDefect1.png", guiShow)
                        eggplantCount = eggplantCount + 1
                        if(eggplantCount > 1 ) : eggplantCount = 0
                                 
                    elif labelName == "tomato" :
                        if tomatoCount == 0:
                           detectCount = classification(labelName, "images/tomatoUnDefect.png", "images/tomatoUnDefect1.png", guiShow)
                        else:
                           detectCount = classification(labelName, "images/tomatoDefect.png", "images/tomatoDefect1.png", guiShow)
                        tomatoCount = tomatoCount + 1
                        if(tomatoCount > 1 ) : tomatoCount = 0
                                                       



                if finalLabel!=labelName and timerOn==False:
                    timerOn = True
                    logUdate(labelName)
                    finalLabel=labelName

                if(categoryVar.get()==1):
                    if labelName == "apple" or labelName == "banana" or labelName == "orange":
                        category = "Fruit"
                    if labelName == "eggplant" or labelName == "carrot" or labelName == "tomato":
                        category = "Vegetable"
                    cv.putText(frame, f'Category: {category}', (380, 50),
                            cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

                label = "%s : %f" % (labelName, score)
                cv.rectangle(frame, box, color, 1)
                cv.putText(frame, label, (box[0], box[1]-10),
                        cv.FONT_HERSHEY_COMPLEX, 0.3, color, 1)                    

            # Use class_label to control
        if timer == 5:
            ser.write(bytes('S', 'UTF-8'))
        if(timer == 6):
            if detectCount == 0 :
                ser.write(bytes('A', 'UTF-8'))
                print("A")
            elif detectCount == 1 :
                ser.write(bytes('B', 'UTF-8'))
                print("B")                             
            elif detectCount > 1 :
                ser.write(bytes('C', 'UTF-8'))
                print("C")                
            detectCount=0         


        if timer > 10:
            timer = 0
            timerOn = False
            finalLabel = 'None'
            ser.write(bytes('R', 'UTF-8'))                
        endingTime = time.time() - starting_time
        fps = frame_counter/endingTime
        # print(fps)
        cv.putText(frame, f'FPS: {fps}', (20, 50),
                cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
        cv.imshow('frame', frame)
        key = cv.waitKey(1)
        if key == ord('q'):
            ser.write(bytes('P', 'UTF-8'))
            cv.destroyAllWindows()
            break
    #cap.release()
    #cv.destroyAllWindows()        

dashboard.configure(bg="#525561")

# ================Background Image ====================
dashboard_backgroundImage = PhotoImage(file="assets\\image_1.png")
bg_imageDashboard = Label(
    dashboard,
    image=dashboard_backgroundImage,
    bg="#525561"
)
bg_imageDashboard.place(x=120, y=28)

# ================ Header Text Left ====================
dashboard_headerText_image_left = PhotoImage(file="assets\\headerText_image.png")
dashboard_headerText_image_label1 = Label(
    bg_imageDashboard,
    image=Login_headerText_image_left,
    bg="#272A37"
)
dashboard_headerText_image_label1.place(x=60, y=45)

dashboard_headerText1 = Label(
    bg_imageDashboard,
    text="Dashboard",
    fg="#FFFFFF",
    font=("yu gothic ui bold", 20 * -1),
    bg="#272A37"
)
dashboard_headerText1.place(x=110, y=45)

headerText_image_label2.place(x=400, y=45)

dashboard_headerText2 = Label(
    bg_imageDashboard,
    anchor="nw",
    text="Select Options",
    fg="#FFFFFF",
    font=("yu gothic ui Bold", 20 * -1),
    bg="#272A37"
)
dashboard_headerText2.place(x=450, y=45)

dashboard_Manual = Button(
    bg_imageDashboard,
    text="Manual Machine Control",
    fg="#206DB4",
    font=("yu gothic ui Bold", 18 * -1),
    bg="#272A37",
    bd=0,
    cursor="hand2",
    activebackground="#272A37",
    activeforeground="#ffffff",
    command=lambda : show_frame(manual)
)
dashboard_Manual.place(x=410, y=90, width=220, height=70)

dashboard_imgP = Button(
    bg_imageDashboard,
    text="Image Processing",
    fg="#206DB4",
    font=("yu gothic ui Bold", 18 * -1),
    bg="#272A37",
    bd=0,
    cursor="hand2",
    activebackground="#272A37",
    activeforeground="#ffffff",
    command=open_camera
)
dashboard_imgP.place(x=410, y=155, width=220, height=70)

dashboard_categoryCheck = Checkbutton(
    bg_imageDashboard,
    text="Category",
    variable=categoryVar, 
    onvalue=1, 
    offvalue=0,
    fg="#206DB4",
    font=("yu gothic ui Bold", 14 * -1),
    bg="#272A37",
    bd=0,
    activebackground="#272A37",
    activeforeground="#ffffff",    
)
dashboard_categoryCheck.place(x=380, y=200, width=100, height=70)  


dashboard_classificationCheck = Checkbutton(
    bg_imageDashboard,
    text="Classification",
    variable=classificationVar, 
    onvalue=1, 
    offvalue=0,
    fg="#206DB4",
    font=("yu gothic ui Bold", 14 * -1),
    bg="#272A37",
    bd=0,
    activebackground="#272A37",
    activeforeground="#ffffff",    
)
dashboard_classificationCheck.place(x=550, y=200, width=120, height=70) 





dashboard_activity = Button(
    bg_imageDashboard,
    text="Activity Logs",
    fg="#206DB4",
    font=("yu gothic ui Bold", 18 * -1),
    bg="#272A37",
    bd=0,
    cursor="hand2",
    activebackground="#272A37",
    activeforeground="#ffffff",
    command=open_logs
)
dashboard_activity.place(x=400, y=250, width=220, height=70)

dashboard_logout = Button(
    bg_imageDashboard,
    text="Logout",
    fg="#FFFFFF",
    font=("yu gothic ui Bold", 18 * -1),
    bg="#272A37",
    bd=0,
    cursor="hand2",
    activebackground="#272A37",
    activeforeground="#ffffff",
    command=lambda : show_frame(sign_in)
)
dashboard_logout.place(x=760, y=520, width=220, height=70)


# ====================================================
# ============ MANUAL MACHINE DASHBOARD ==============
# ====================================================

def set_button1_start():
    ser.write(bytes('S', 'UTF-8'))
    print('Motor Start')

def set_button1_stop():
    ser.write(bytes('P', 'UTF-8'))
    print('Motor Stop')  

choices = ['COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'COM10', 'COM11', 'COM12']
manual.configure(bg="#525561")

# ================Background Image ====================
manual_backgroundImage = PhotoImage(file="assets\\image_1.png")
bg_imageManual = Label(
    manual,
    image=manual_backgroundImage,
    bg="#525561"
)
bg_imageManual.place(x=120, y=28)

# ================ Header Text Left ====================
manual_headerText_image_left = PhotoImage(file="assets\\headerText_image.png")
manual_headerText_image_label1 = Label(
    bg_imageManual,
    image=Login_headerText_image_left,
    bg="#272A37"
)
manual_headerText_image_label1.place(x=60, y=45)

manual_headerText1 = Label(
    bg_imageManual,
    text="Maunal Machine Control",
    fg="#FFFFFF",
    font=("yu gothic ui bold", 20 * -1),
    bg="#272A37"
)
manual_headerText1.place(x=110, y=45)

headerText_image_label2.place(x=400, y=45)

manual_headerText2 = Label(
    bg_imageManual,
    anchor="nw",
    text="Select COM port",
    fg="#FFFFFF",
    font=("yu gothic ui Bold", 20 * -1),
    bg="#272A37"
)
manual_headerText2.place(x=450, y=45)

variable = StringVar(bg_imageManual)
variable.set('COM1')

combo1 = ttk.Combobox(bg_imageManual, values=choices)
combo1.place(x=460, y=85)

manual_back = Button(
    bg_imageManual,
    text="Back",
    fg="#FFFFFF",
    font=("yu gothic ui Bold", 18 * -1),
    bg="#272A37",
    bd=0,
    cursor="hand2",
    activebackground="#272A37",
    activeforeground="#ffffff",
    command=lambda : show_frame(dashboard)
)
manual_back.place(x=760, y=500, width=220, height=70)

# manual_connect = Button(
#     bg_imageManual,
#     text="Connect",
#     fg="#206DB4",
#     font=("yu gothic ui Bold", 18 * -1),
#     bg="#272A37",
#     bd=0,
#     cursor="hand2",
#     activebackground="#272A37",
#     activeforeground="#ffffff",
#     command=lambda : show_frame(manual)
# )
# manual_connect.place(x=410, y=125, width=220, height=70)

manual_start = Button(
    bg_imageManual,
    text="Start",
    fg="#206DB4",
    font=("yu gothic ui Bold", 18 * -1),
    bg="#272A37",
    bd=0,
    cursor="hand2",
    activebackground="#272A37",
    activeforeground="#ffffff",
    command=set_button1_start
)
manual_start.place(x=410, y=230, width=220, height=70)

manual_stop = Button(
    bg_imageManual,
    text="Stop",
    fg="#206DB4",
    font=("yu gothic ui Bold", 18 * -1),
    bg="#272A37",
    bd=0,
    cursor="hand2",
    activebackground="#272A37",
    activeforeground="#ffffff",
    command=set_button1_stop
)
manual_stop.place(x=410, y=320, width=220, height=70)

main.resizable(False, False)
main.mainloop()