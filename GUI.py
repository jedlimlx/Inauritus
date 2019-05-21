from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk

import cv2, os
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def classify(image_path):
    # Read the image_data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line in tf.gfile.GFile("logs/trained_labels.txt")]

    # Unpersists graph from file
    with tf.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        human_string = label_lines[top_k[0]]
        return human_string

global imgnum
imgnum = 0
x = 400
y = 150
h = 200
w = 200

splashScreen = True
if splashScreen:
    root = Tk()
    root.overrideredirect(True)
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    root.geometry('%dx%d+%d+%d' % (width*0.8, height*0.8, width*0.1, height*0.1))
    image_file = "Inauritus.gif"
    image = PhotoImage(file=image_file)
    canvas = Canvas(root, height=height*0.8, width=width*0.8, bg="black")
    canvas.create_image(width*0.8/2, height*0.8/2, image=image)
    canvas.pack()
    root.after(5000, root.destroy)
    root.mainloop()

width, height = 1000, 750
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

class MultiImage():
    def __init__(self, tab):
        self.scrollbar = Scrollbar(tab, orient=VERTICAL)
        self.scrollbar.pack(side=RIGHT, fill=Y)
        self.c = Canvas(tab, yscrollcommand=self.scrollbar.set)
        self.c.pack(expand=YES, fill=BOTH)
        self.c.config(scrollregion=(0, 0, 5000, 5000))
        self.scrollbar.config(command=self.c.yview)
        self.imglst = []
        
    def add_image(self, img, n):
        self.image = self.c.create_image((n % 8) * 225 + 125,
                                         125 + (n // 8) * 225, image=img)
        self.image = img
        self.imglst.append(self.image)
        print(n)

    def clear(self):
        self.c.delete("all")
        self.imglst = []
        text.set("")

def translate():
    global text
    global imgnum
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = cv2image[y:y+h, x:x+w]
    gray_img = gray[y:y+h, x:x+w]
    img_to_predict = cv2.resize(gray_img, (28, 28), interpolation = cv2.INTER_AREA)
    '''
    try:
        predict_input = np.array(img_to_predict).reshape(-1, 4096)
        predictions = Agent.predict(predict_input)
        print(predict_input)
    except:
        predict_input = np.array(img_to_predict).\
                         reshape(-1, 28, 28, 1).astype('float32') / 255
        predictions = Agent.predict(predict_input)
        print(predict_input)
    print(predictions)
    letter = Agent.convert_prediction_array_to_output(predictions)
    '''
    cv2.imwrite("predict.jpg", img_to_predict)
    letter = classify("predict.jpg")
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(roi, letter,(5, 190), font, 2, (0), 4, cv2.LINE_AA)
    imgtk = opencv2tkinter(roi)
    #print(roi)
    TranslateLabel.config(text = "Translates to: " + letter)
    if "nothing" in letter:
        pass
    elif "del" in letter:
        MI.add_image(imgtk, imgnum)
        text.set(text.get()[:-1])
    elif "space" in letter:
        MI.add_image(imgtk, imgnum)
        text.set(text.get() + " ")
    else:
        MI.add_image(imgtk, imgnum)
        text.set(text.get() + letter)
    WordLabel.config(text=text)
    imgnum += 1

def opencv2tkinter(img):
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    return imgtk

def clear():
    global MI
    global imgnum
    text.set('')
    MI.clear()
    imgnum = 0

def show_frame():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.rectangle(cv2image,(x,y),(x+w,y+h),(255, 0, 0, 1),2)
    imgtk = opencv2tkinter(cv2image)
    pic.imgtk = imgtk
    pic.configure(image=imgtk)
    pic.after(5, show_frame)

root = Tk()
text = StringVar()
text.set("")
#root.bind('<Escape>', lambda e: root.quit())
root.title("Inauritus")
root.state("zoomed")
imgicon = PhotoImage(file='InauritusBlack.gif')
root.tk.call('wm', 'iconphoto', root._w, imgicon)

global tabControl
tabControl = ttk.Notebook()
TabSign = ttk.Frame(tabControl)
TabImg = ttk.Frame(tabControl)
TabSentence = ttk.Frame(tabControl)
TabHelp = ttk.Frame(tabControl)

tabControl.add(TabSign, text='Sign Language Translator')
tabControl.add(TabImg, text='Captures')
tabControl.add(TabSentence, text='Sentence')
tabControl.add(TabHelp, text='Help')
tabControl.pack(expand=1, fill="both")

pic = Label(TabSign)
pic.pack(padx = 30, pady = 30)
InfoLabel = Label(TabSign, text = "Place your hand in the White Box", font=("Helvetica", 20))
InfoLabel.pack(padx = 30)

TranslateButton = Button(TabSign, text = "Press to Translate", font=("Helvetica", 20), command=translate)
TranslateButton.pack(padx = 30, pady = 10)

TranslateLabel = Label(TabSign, text = "Translates to: ", font=("Helvetica", 20))
TranslateLabel.pack(padx = 30)

Placeholder = Label(TabSign, text = "")
Placeholder.pack()

global MI
MI = MultiImage(TabImg)

WordLabel = Label(TabSentence, font=("Helvetica", 20), textvariable=text)
WordLabel.pack()

ClearButton = Button(TabSentence, text = "Clear", font=("Helvetica", 20), command=clear)
ClearButton.pack(pady=10)

ASL = Label(TabHelp)
ASL.pack()
img = Image.open("ASL.png")
img = img.resize((480,640), Image.ANTIALIAS)
photoImg = ImageTk.PhotoImage(img)
ASL.config(image=photoImg)

HelpText = Text(TabHelp, wrap=WORD, height=100, width=100)
HelpText.pack()
file = open("README.txt", 'r')
README = file.read()
file.close()
HelpText.insert(END, README)
HelpText.config(state=DISABLED)

show_frame()
root.mainloop()
