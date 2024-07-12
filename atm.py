import tkinter as tk
import time
import cv2
import numpy as np
import copy
import math
import imutils
import pyautogui
import time
from PIL import Image,ImageTk

#import mouse_events_control as mec

from tensorflow.keras.models import load_model


model=load_model("handmodel_fingers_model.h5")




current_balance = 1000

class SampleApp(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.shared_data = {'Balance':tk.IntVar()}

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)

        self.frames = {}
        for F in (StartPage, MenuPage, WithdrawPage, DepositPage, BalancePage):
            page_name = F.__name__
            frame = F(parent=container, controller=self,width=516,height=760)
            self.frames[page_name] = frame

            # put all of the pages in the same location;
            # the one on the top of the stacking order
            # will be the one that is visible.
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")
        sideFrame=tk.Frame(container,bd=7,width=10,height=760,relief=tk.RIDGE)
        sideFrame.grid(row=0,column=1,padx=2)
        L1=tk.Label(sideFrame)
        L1.grid(row=0,column=0)
        L2=tk.Label(sideFrame)
        L2.grid(row=1,column=0)
        
        cap_region_x_begin=0.5  # start point/total width
        cap_region_y_end=0.8  # start point/total width
        threshold = 60  #  BINARY threshold
        blurValue = 41  # GaussianBlur parameter
        bgSubThreshold = 50
        self.learningRate = 0
        
        top, right, bottom, left = 50, 350, 300, 650

        # variables
        isBgCaptured = 0   # bool, whether the background captured
        triggerSwitch = False  # if true, keyborad simulator works

        lap_width,lap_height=pyautogui.size()

        self.bgModel = None
        # Camera
        camera = cv2.VideoCapture(0)
        camera.set(10,200)
        cv2.namedWindow('trackbar')
        cv2.createTrackbar('trh1', 'trackbar', threshold, 100, self.printThreshold)
        first_itr=0
        while camera.isOpened():
            ret, frame = camera.read()
            threshold =cv2.getTrackbarPos('trh1', 'trackbar')
            frame = imutils.resize(frame, width=700)
            frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
            frame = cv2.flip(frame, 1)  # flip the frame horizontally
            width,height,a=frame.shape
            upper_left = (width // 4, height // 4)
            bottom_right = (width * 3 // 4, height * 3 // 4)
            cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
            #cv2.rectangle(frame, upper_left, bottom_right, (0, 255, 0), thickness=1)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img=ImageTk.PhotoImage(Image.fromarray(img))
            L1['image']=img
            cv2.imshow('original', frame)
            #if first_itr==0:
                #cv2.imshow('original', frame)
            #elif first_itr==1:
                #cv2.destroyWindow('original')
                #first_itr+=1
            #  Main operation
            container.update()
            if isBgCaptured == 1:  # this part wont run until background captured
                img = self.removeBG(frame)
                img = img[top:bottom, right:left]  # clip the ROI
                img_height, img_width, c = img.shape
                if first_itr==0:cv2.imshow('mask', img)
                elif first_itr==1:cv2.destroyWindow('mask')
                # convert the image into binary image
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
                if first_itr==0:cv2.imshow('blur', blur)
                elif first_itr==1:cv2.destroyWindow('blur')
                ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
                if first_itr==0:cv2.imshow('ori', thresh)
                elif first_itr==1:cv2.destroyWindow('ori')

                # get the coutours
                thresh1 = copy.deepcopy(thresh)
                contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                #cnt=contours[0]
                #x,y,w,h = cv2.boundingRect(cnt)
                length = len(contours)
                maxArea = -1
                if length > 0:
                    for i in range(length):  # find the biggest contour (according to area)
                        temp = contours[i]
                        area = cv2.contourArea(temp)
                        if area > maxArea:
                            maxArea = area
                            ci = i

                    res = contours[ci]
                    #print(res[0][0])
                    extTop=(res[0][0][0],res[0][0][1])
                    hull = cv2.convexHull(res)
                    drawing = np.zeros(img.shape, np.uint8)
                    cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
                    cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)
                    #cv2.rectangle(drawing,(x,y),(x+w,y+h),(0,255,0),2)
                    cv2.circle(drawing, extTop, 8, (255, 0, 0), -1)

                    fingers=self.count(thresh)
                    mov_x,mov_y=pyautogui.position()
                    move_pixel=15
                    if fingers==1 and 0<=mov_x<lap_width and 0<=(mov_y+move_pixel)<lap_height:
                        #pyautogui.moveTo(mov_x,mov_y)
                        pyautogui.moveRel(0,move_pixel)
                    if fingers==2:
                        pyautogui.click()
                        time.sleep(0.5)
                    if fingers==3 and 0<=mov_x<lap_width and 0<=(mov_y-move_pixel)<lap_height:
                        pyautogui.moveRel(0,-move_pixel)
                    if fingers==4 and 0<=(mov_x+move_pixel)<lap_width and 0<=(mov_y)<lap_height:
                        pyautogui.moveRel(move_pixel,0)
                    if fingers==5 and 0<=(mov_x-move_pixel)<lap_width and 0<=(mov_y+move_pixel)<lap_height:
                        pyautogui.moveRel(-move_pixel,0)
                    cv2.putText(drawing, str(fingers), (200, 45), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,255), 2)
                    cv2.putText(frame, str(fingers), (200, 45), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,255), 2)
                    

                #cv2.imshow('output', drawing)
                out_img=ImageTk.PhotoImage(Image.fromarray(drawing))
                L2['image']=out_img

            # Keyboard OP
            k = cv2.waitKey(10)
            if k == 27:  # press ESC to exit
                camera.release()
                cv2.destroyAllWindows()
                break
            elif k == ord('b'):  # press 'b' to capture the background
                self.bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
                isBgCaptured = 1
                first_itr+=1
                print( '!!!Background Captured!!!')
            elif k == ord('r'):  # press 'r' to reset the background
                self.bgModel = None
                triggerSwitch = False
                isBgCaptured = 0
                print ('!!!Reset BackGround!!!')
            elif k == ord('n'):
                triggerSwitch = True
                print ('!!!Trigger On!!!')
         

    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        frame = self.frames[page_name]
        frame.tkraise()
    
    def printThreshold(self,thr):
        print("! Changed threshold to "+str(thr))


    def removeBG(self,frame):
        fgmask = self.bgModel.apply(frame,learningRate=self.learningRate)
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        kernel = np.ones((3, 3), np.uint8)
        fgmask = cv2.erode(fgmask, kernel, iterations=1)
        res = cv2.bitwise_and(frame, frame, mask=fgmask)
        return res
    def count(self,thresholded):
    
        thresholded = cv2.resize(thresholded,(128,128))
        thresholded = thresholded.reshape(-1,128,128,1).astype('float32')
        thresholded = thresholded / 255
        prob = model.predict(thresholded)
        res = np.argmax(prob)
        return res


class StartPage(tk.Frame):

    def __init__(self, parent, controller,width,height):
        tk.Frame.__init__(self, parent,bg='#3d3d5c' ,width=width, height=height)
        self.controller = controller

        self.controller.title('ATM')
        self.controller.state('zoomed')
        #self.controller.iconphoto(False,tk.PhotoImage(file='C:/Users/urban boutique/Documents/atm tutorial/atm.png'))

        heading_label = tk.Label(self,
                                                     text='TOUCHLESS ATM',
                                                     font=('orbitron',45,'bold'),
                                                     foreground='#ffffff',
                                                     background='#3d3d5c')
        heading_label.pack(pady=25)

        space_label = tk.Label(self,height=4,bg='#3d3d5c')
        space_label.pack()

        password_label = tk.Label(self,
                                                      text='Enter your password',
                                                      font=('orbitron',13),
                                                      bg='#3d3d5c',
                                                      fg='white')
        password_label.pack(pady=10)

        
        password_entry_box = tk.Text(self,
                                                              
                                                              font=('orbitron',12),
                                                              width=22,height=1
                                        )
        password_entry_box.focus_set()
        password_entry_box.pack(pady=7)

        #def handle_focus_in(_):
            #password_entry_box.configure(fg='black',show='*')
            
        #password_entry_box.bind('<FocusIn>',handle_focus_in)

        def check_password():
           pinNo=password_entry_box.get("1.0","end-1c")
           if  pinNo== str("2213"):
               password_entry_box.delete("1.0",tk.END)
               incorrect_password_label['text']=''
               controller.show_frame('MenuPage')
           else:
               incorrect_password_label['text']='Incorrect Password'
                

        incorrect_password_label = tk.Label(self,
                                                                        text='',
                                                                        font=('orbitron',13),
                                                                        fg='white',
                                                                        bg='#33334d',
                                                                        anchor='n')
        incorrect_password_label.pack(fill='both',expand=True)
        
        button_frame = tk.Frame(self,bg='#33334d')
        button_frame.grid_rowconfigure(4, weight=1)
        button_frame.grid_columnconfigure(4, weight=1)
        button_frame.pack(fill='both',expand=True,padx=250,pady=35)
        
        def clear():
            password_entry_box.delete("1.0",tk.END)
            incorrect_password_label['text']=''
        
        def insert0():
            value0=0
            password_entry_box.insert(tk.END,value0)
        def insert1():
            value1=1
            password_entry_box.insert(tk.END,value1)
        def insert2():
            value2=2
            password_entry_box.insert(tk.END,value2)
        def insert3():
            value3=3
            password_entry_box.insert(tk.END,value3)
        def insert4():
            value4=4
            password_entry_box.insert(tk.END,value4)
        def insert5():
            value5=5
            password_entry_box.insert(tk.END,value5)
        def insert6():
            value6=6
            password_entry_box.insert(tk.END,value6)
        def insert7():
            value7=7
            password_entry_box.insert(tk.END,value7)
        def insert8():
            value8=8
            password_entry_box.insert(tk.END,value8)
        def insert9():
            value9=9
            password_entry_box.insert(tk.END,value9)
        
        self.img1=tk.PhotoImage(file="one.png")
        btn1=tk.Button(button_frame,width=90,height=50,command=insert1,
                              image=self.img1).grid(row=0,column=0,padx=5)
        
        
        self.img2=tk.PhotoImage(file="two.png")
        btn2=tk.Button(button_frame,width=90,height=50,command=insert2,
                              image=self.img2).grid(row=0,column=1,padx=5)
        
        
        self.img3=tk.PhotoImage(file="three.png")
        btn3=tk.Button(button_frame,width=90,height=50,command=insert3,
                              image=self.img3).grid(row=0,column=2,padx=5)
        
        self.imgCE=tk.PhotoImage(file="cancel.png")
        btnCancel=tk.Button(button_frame,width=90,height=50,
                              image=self.imgCE).grid(row=0,column=3,padx=5)
        #==============
        self.img4=tk.PhotoImage(file="four.png")
        btn4=tk.Button(button_frame,width=90,height=50,command=insert4,
                              image=self.img4).grid(row=1,column=0,padx=5,pady=5)
        
        self.img5=tk.PhotoImage(file="five.png")
        btn5=tk.Button(button_frame,width=90,height=50,command=insert5,
                              image=self.img5).grid(row=1,column=1,padx=5,pady=5)
        
        self.img6=tk.PhotoImage(file="six.png")
        btn6=tk.Button(button_frame,width=90,height=50,command=insert6,
                              image=self.img6).grid(row=1,column=2,padx=5,pady=5)
        
        self.imgCL=tk.PhotoImage(file="clear.png")
        btnClear=tk.Button(button_frame,width=90,height=50,command=clear,
                              image=self.imgCL).grid(row=1,column=3,padx=5,pady=5)
        #==============
        self.img7=tk.PhotoImage(file="seven.png")
        btn7=tk.Button(button_frame,width=90,height=50,command=insert7,
                              image=self.img7).grid(row=2,column=0,padx=5,pady=5)
        
        self.img8=tk.PhotoImage(file="eight.png")
        btn8=tk.Button(button_frame,width=90,height=50,command=insert8,
                              image=self.img8).grid(row=2,column=1,padx=5,pady=5)
        
        self.img9=tk.PhotoImage(file="nine.png")
        btn9=tk.Button(button_frame,width=90,height=50,command=insert9,
                              image=self.img9).grid(row=2,column=2,padx=5,pady=5)
        
        self.imgEnter=tk.PhotoImage(file="enter.png")
        btnEnter=tk.Button(button_frame,width=90,height=50,command=check_password,
                              image=self.imgEnter).grid(row=2,column=3,padx=5,pady=5)
        #==============
        
        self.imgSp1=tk.PhotoImage(file="empty.png")
        btnsp1=tk.Button(button_frame,width=90,height=50,
                              image=self.imgSp1).grid(row=3,column=0,padx=5,pady=5)
        
        self.img0=tk.PhotoImage(file="zero.png")
        self.btn0=tk.Button(button_frame,width=90,height=50,command=insert0,
                              image=self.img0).grid(row=3,column=1,padx=5,pady=5)
        
        #self.imgSp2=PhotoImage(file="empty.png")
        btnsp2=tk.Button(button_frame,width=90,height=50,
                              image=self.imgSp1).grid(row=3,column=2,padx=5,pady=5)
        
        #self.imgEnter=PhotoImage(file="enter.png")
        btnsp3=tk.Button(button_frame,width=90,height=50,
                              image=self.imgSp1).grid(row=3,column=3,padx=5,pady=5)
        
        
        
        bottom_frame = tk.Frame(self,relief='raised',borderwidth=3)
        bottom_frame.pack(fill='x',side='bottom')

        visa_photo = tk.PhotoImage(file='visa.png')
        visa_label = tk.Label(bottom_frame,image=visa_photo)
        visa_label.pack(side='left')
        visa_label.image = visa_photo

        mastercard_photo = tk.PhotoImage(file='mastercard.png')
        mastercard_label = tk.Label(bottom_frame,image=mastercard_photo)
        mastercard_label.pack(side='left')
        mastercard_label.image = mastercard_photo

        american_express_photo = tk.PhotoImage(file='american-express.png')
        american_express_label = tk.Label(bottom_frame,image=american_express_photo)
        american_express_label.pack(side='left')
        american_express_label.image = american_express_photo

        def tick():
            current_time = time.strftime('%I:%M %p').lstrip('0').replace(' 0',' ')
            time_label.config(text=current_time)
            time_label.after(200,tick)
            
        time_label = tk.Label(bottom_frame,font=('orbitron',12))
        time_label.pack(side='right')

        tick()
        


class MenuPage(tk.Frame):

    def __init__(self, parent, controller,width,height):
        tk.Frame.__init__(self, parent,bg='#3d3d5c',width=width, height=height)
        self.controller = controller
   
        heading_label = tk.Label(self,
                                                     text='TOUCHLESS ATM',
                                                     font=('orbitron',45,'bold'),
                                                     foreground='#ffffff',
                                                     background='#3d3d5c')
        heading_label.pack(pady=25)

        main_menu_label = tk.Label(self,
                                                           text='Main Menu',
                                                           font=('orbitron',13),
                                                           fg='white',
                                                           bg='#3d3d5c')
        main_menu_label.pack()

        selection_label = tk.Label(self,
                                                           text='Please make a selection',
                                                           font=('orbitron',13),
                                                           fg='white',
                                                           bg='#3d3d5c',
                                                           anchor='w')
        selection_label.pack(fill='x')

        button_frame = tk.Frame(self,bg='#33334d')
        button_frame.pack(fill='both',expand=True)

        def withdraw():
            controller.show_frame('WithdrawPage')
            
        withdraw_button = tk.Button(button_frame,
                                                            text='Withdraw',
                                                            command=withdraw,
                                                            relief='raised',
                                                            borderwidth=3,
                                                            width=50,
                                                            height=5)
        withdraw_button.grid(row=0,column=0,pady=5)
        

        def deposit():
            controller.show_frame('DepositPage')
            
        deposit_button = tk.Button(button_frame,
                                                            text='Deposit',
                                                            command=deposit,
                                                            relief='raised',
                                                            borderwidth=3,
                                                            width=50,
                                                            height=5)
        deposit_button.grid(row=1,column=0,pady=5)

        def balance():
            controller.show_frame('BalancePage')
            
        balance_button = tk.Button(button_frame,
                                                            text='Balance',
                                                            command=balance,
                                                            relief='raised',
                                                            borderwidth=3,
                                                            width=50,
                                                            height=5)
        balance_button.grid(row=2,column=0,pady=5)

        def exit():
            controller.show_frame('StartPage')
            
        exit_button = tk.Button(button_frame,
                                                            text='Exit',
                                                            command=exit,
                                                            relief='raised',
                                                            borderwidth=3,
                                                            width=50,
                                                            height=5)
        exit_button.grid(row=3,column=0,pady=5)


        bottom_frame = tk.Frame(self,relief='raised',borderwidth=3)
        bottom_frame.pack(fill='x',side='bottom')

        visa_photo = tk.PhotoImage(file='visa.png')
        visa_label = tk.Label(bottom_frame,image=visa_photo)
        visa_label.pack(side='left')
        visa_label.image = visa_photo

        mastercard_photo = tk.PhotoImage(file='mastercard.png')
        mastercard_label = tk.Label(bottom_frame,image=mastercard_photo)
        mastercard_label.pack(side='left')
        mastercard_label.image = mastercard_photo

        american_express_photo = tk.PhotoImage(file='american-express.png')
        american_express_label = tk.Label(bottom_frame,image=american_express_photo)
        american_express_label.pack(side='left')
        american_express_label.image = american_express_photo

        def tick():
            current_time = time.strftime('%I:%M %p').lstrip('0').replace(' 0',' ')
            time_label.config(text=current_time)
            time_label.after(200,tick)
            
        time_label = tk.Label(bottom_frame,font=('orbitron',12))
        time_label.pack(side='right')

        tick()


class WithdrawPage(tk.Frame):
    
    def __init__(self, parent, controller,width,height):
        tk.Frame.__init__(self, parent,bg='#3d3d5c',width=width, height=height)
        self.controller = controller


        heading_label = tk.Label(self,
                                                     text='TOUCHLESS ATM',
                                                     font=('orbitron',45,'bold'),
                                                     foreground='#ffffff',
                                                     background='#3d3d5c')
        heading_label.pack(pady=25)

        choose_amount_label = tk.Label(self,
                                                           text='Choose the amount you want to withdraw',
                                                           font=('orbitron',13),
                                                           fg='white',
                                                           bg='#3d3d5c')
        choose_amount_label.pack()

        button_frame = tk.Frame(self,bg='#33334d')
        button_frame.pack(fill='both',expand=True)

        def withdraw(amount):
            global current_balance
            current_balance -= amount
            controller.shared_data['Balance'].set(current_balance)
            controller.show_frame('MenuPage')
            
        twenty_button = tk.Button(button_frame,
                                                       text='20',
                                                       command=lambda:withdraw(20),
                                                       relief='raised',
                                                       borderwidth=3,
                                                       width=50,
                                                       height=5)
        twenty_button.grid(row=0,column=0,pady=5)

        forty_button = tk.Button(button_frame,
                                                       text='40',
                                                       command=lambda:withdraw(40),
                                                       relief='raised',
                                                       borderwidth=3,
                                                       width=50,
                                                       height=5)
        forty_button.grid(row=1,column=0,pady=5)

        sixty_button = tk.Button(button_frame,
                                                       text='60',
                                                       command=lambda:withdraw(60),
                                                       relief='raised',
                                                       borderwidth=3,
                                                       width=50,
                                                       height=5)
        sixty_button.grid(row=2,column=0,pady=5)

        eighty_button = tk.Button(button_frame,
                                                       text='80',
                                                       command=lambda:withdraw(80),
                                                       relief='raised',
                                                       borderwidth=3,
                                                       width=50,
                                                       height=5)
        eighty_button.grid(row=3,column=0,pady=5)

        one_hundred_button = tk.Button(button_frame,
                                                       text='100',
                                                       command=lambda:withdraw(100),
                                                       relief='raised',
                                                       borderwidth=3,
                                                       width=50,
                                                       height=5)
        one_hundred_button.grid(row=0,column=1,pady=5)

        two_hundred_button = tk.Button(button_frame,
                                                       text='200',
                                                       command=lambda:withdraw(200),
                                                       relief='raised',
                                                       borderwidth=3,
                                                       width=50,
                                                       height=5)
        two_hundred_button.grid(row=1,column=1,pady=5)

        three_hundred_button = tk.Button(button_frame,
                                                       text='300',
                                                       command=lambda:withdraw(300),
                                                       relief='raised',
                                                       borderwidth=3,
                                                       width=50,
                                                       height=5)
        three_hundred_button.grid(row=2,column=1,pady=5)

        cash = tk.StringVar()
        other_amount_entry = tk.Entry(button_frame,
                                                              textvariable=cash,
                                                              width=59,
                                                              justify='right')
        other_amount_entry.grid(row=3,column=1,pady=5,ipady=30)

        def other_amount(_):
            global current_balance
            current_balance -= int(cash.get())
            controller.shared_data['Balance'].set(current_balance)
            cash.set('')
            controller.show_frame('MenuPage')
            
        other_amount_entry.bind('<Return>',other_amount)

        bottom_frame = tk.Frame(self,relief='raised',borderwidth=3)
        bottom_frame.pack(fill='x',side='bottom')

        visa_photo = tk.PhotoImage(file='visa.png')
        visa_label = tk.Label(bottom_frame,image=visa_photo)
        visa_label.pack(side='left')
        visa_label.image = visa_photo

        mastercard_photo = tk.PhotoImage(file='mastercard.png')
        mastercard_label = tk.Label(bottom_frame,image=mastercard_photo)
        mastercard_label.pack(side='left')
        mastercard_label.image = mastercard_photo

        american_express_photo = tk.PhotoImage(file='american-express.png')
        american_express_label = tk.Label(bottom_frame,image=american_express_photo)
        american_express_label.pack(side='left')
        american_express_label.image = american_express_photo

        def tick():
            current_time = time.strftime('%I:%M %p').lstrip('0').replace(' 0',' ')
            time_label.config(text=current_time)
            time_label.after(200,tick)
            
        time_label = tk.Label(bottom_frame,font=('orbitron',12))
        time_label.pack(side='right')

        tick()
   

class DepositPage(tk.Frame):
    
    def __init__(self, parent, controller,width,height):
        tk.Frame.__init__(self, parent,bg='#3d3d5c',width=width, height=height)
        self.controller = controller

        heading_label = tk.Label(self,
                                                     text='TOUCHLESS ATM',
                                                     font=('orbitron',45,'bold'),
                                                     foreground='#ffffff',
                                                     background='#3d3d5c')
        heading_label.pack(pady=25)

        space_label = tk.Label(self,height=4,bg='#3d3d5c')
        space_label.pack()

        enter_amount_label = tk.Label(self,
                                                      text='Enter amount',
                                                      font=('orbitron',13),
                                                      bg='#3d3d5c',
                                                      fg='white')
        enter_amount_label.pack(pady=10)

        cash = tk.StringVar()
        deposit_entry = tk.Entry(self,
                                                  textvariable=cash,
                                                  font=('orbitron',12),
                                                  width=22)
        deposit_entry.pack(ipady=7)

        def deposit_cash():
            global current_balance
            current_balance += int(cash.get())
            controller.shared_data['Balance'].set(current_balance)
            controller.show_frame('MenuPage')
            cash.set('')
            
        enter_button = tk.Button(self,
                                                     text='Enter',
                                                     command=deposit_cash,
                                                     relief='raised',
                                                     borderwidth=3,
                                                     width=40,
                                                     height=3)
        enter_button.pack(pady=10)

        two_tone_label = tk.Label(self,bg='#33334d')
        two_tone_label.pack(fill='both',expand=True)

        bottom_frame = tk.Frame(self,relief='raised',borderwidth=3)
        bottom_frame.pack(fill='x',side='bottom')

        visa_photo = tk.PhotoImage(file='visa.png')
        visa_label = tk.Label(bottom_frame,image=visa_photo)
        visa_label.pack(side='left')
        visa_label.image = visa_photo

        mastercard_photo = tk.PhotoImage(file='mastercard.png')
        mastercard_label = tk.Label(bottom_frame,image=mastercard_photo)
        mastercard_label.pack(side='left')
        mastercard_label.image = mastercard_photo

        american_express_photo = tk.PhotoImage(file='american-express.png')
        american_express_label = tk.Label(bottom_frame,image=american_express_photo)
        american_express_label.pack(side='left')
        american_express_label.image = american_express_photo

        def tick():
            current_time = time.strftime('%I:%M %p').lstrip('0').replace(' 0',' ')
            time_label.config(text=current_time)
            time_label.after(200,tick)
            
        time_label = tk.Label(bottom_frame,font=('orbitron',12))
        time_label.pack(side='right')

        tick()


class BalancePage(tk.Frame):
    
    def __init__(self, parent, controller,width,height):
        tk.Frame.__init__(self, parent,bg='#3d3d5c',width=width, height=height)
        self.controller = controller

        
        heading_label = tk.Label(self,
                                                     text='TOUCHLESS ATM',
                                                     font=('orbitron',45,'bold'),
                                                     foreground='#ffffff',
                                                     background='#3d3d5c')
        heading_label.pack(pady=25)

        global current_balance
        controller.shared_data['Balance'].set(current_balance)
        balance_label = tk.Label(self,
                                                  textvariable=controller.shared_data['Balance'],
                                                  font=('orbitron',13),
                                                  fg='white',
                                                  bg='#3d3d5c',
                                                  anchor='w')
        balance_label.pack(fill='x')

        button_frame = tk.Frame(self,bg='#33334d')
        button_frame.pack(fill='both',expand=True)

        def menu():
            controller.show_frame('MenuPage')
            
        menu_button = tk.Button(button_frame,
                                                    command=menu,
                                                    text='Menu',
                                                    relief='raised',
                                                    borderwidth=3,
                                                    width=50,
                                                    height=5)
        menu_button.grid(row=0,column=0,pady=5)

        def exit():
            controller.show_frame('StartPage')
            
        exit_button = tk.Button(button_frame,
                                                 text='Exit',
                                                 command=exit,
                                                 relief='raised',
                                                 borderwidth=3,
                                                 width=50,
                                                 height=5)
        exit_button.grid(row=1,column=0,pady=5)

        bottom_frame = tk.Frame(self,relief='raised',borderwidth=3)
        bottom_frame.pack(fill='x',side='bottom')

        visa_photo = tk.PhotoImage(file='visa.png')
        visa_label = tk.Label(bottom_frame,image=visa_photo)
        visa_label.pack(side='left')
        visa_label.image = visa_photo

        mastercard_photo = tk.PhotoImage(file='mastercard.png')
        mastercard_label = tk.Label(bottom_frame,image=mastercard_photo)
        mastercard_label.pack(side='left')
        mastercard_label.image = mastercard_photo

        american_express_photo = tk.PhotoImage(file='american-express.png')
        american_express_label = tk.Label(bottom_frame,image=american_express_photo)
        american_express_label.pack(side='left')
        american_express_label.image = american_express_photo

        def tick():
            current_time = time.strftime('%I:%M %p').lstrip('0').replace(' 0',' ')
            time_label.config(text=current_time)
            time_label.after(200,tick)
            
        time_label = tk.Label(bottom_frame,font=('orbitron',12))
        time_label.pack(side='right')

        tick()


if __name__ == "__main__":
    app = SampleApp()
    app.mainloop()
