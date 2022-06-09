import tkinter
from tkinter import filedialog
import cv2 as cv
import argparse
import numpy as np
import os.path
import sys
import random
import matplotlib.pyplot as plt #グラフ描画ライブラリ
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from functools import partial
from matplotlib.backend_bases import key_press_handler
import mplcursors

#tkinterオブジェクト生成
root = tkinter.Tk()
root.title("Title") #GUIタイトル
        
#graphの設定
fig,ax1 = plt.subplots()
fig.gca().set_aspect('equal', adjustable = 'box')
#Canvas設定
Canvas = FigureCanvasTkAgg(fig, master = root) #Canvasにfigを追加
Img_Masks = []
Img_boxes = []
Img = []
Img_Mask = []
Img_Mask_mod = []
selectId = -1

# Initialize the parameters
confThreshold = 0.5  # Confidence threshold
maskThreshold = 0.3  # Mask threshold

parser = argparse.ArgumentParser(description='Use this script to run Mask-RCNN object detection and segmentation')
# parser.add_argument('--image', help='Path to image file')
parser.add_argument("--device", default="cpu", help="Device to inference on")
args = parser.parse_args()

# Draw the predicted bounding box, colorize and show the mask on the image
def drawBox(frame, classId, conf, left, top, right, bottom, classMask):
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    
    # Print a label of class.
    label = '%.2f' % conf
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)
    
    # Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

    # Resize the mask, threshold, color and apply it on the image
    classMask = cv.resize(classMask, (right - left + 1, bottom - top + 1))
    mask = (classMask > maskThreshold)
    roi = frame[top:bottom+1, left:right+1][mask]

    colorIndex = random.randint(0, len(colors)-1)
    color = colors[colorIndex]

    frame[top:bottom+1, left:right+1][mask] = ([0.3*color[0], 0.3*color[1], 0.3*color[2]] + 0.7 * roi).astype(np.uint8)

    # Draw the contours on the image
    mask = mask.astype(np.uint8)
    contours, hierarchy = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(frame[top:bottom+1, left:right+1], contours, -1, color, 3, cv.LINE_8, hierarchy, 100)

def postprocess01(frame, boxes, masks):
    # Output size of masks is NxCxHxW where
    # N - number of detected boxes
    # C - number of classes (excluding background)
    # HxW - segmentation shape
    numClasses = masks.shape[1]
    numDetections = boxes.shape[2]

    frameH = frame.shape[0]
    frameW = frame.shape[1]

    for i in range(numDetections):
        box = boxes[0, 0, i]
        mask = masks[i]
        score = box[2]
        if score > confThreshold:
            classId = int(box[1])
            
            # Extract the bounding box
            left = int(frameW * box[3])
            top = int(frameH * box[4])
            right = int(frameW * box[5])
            bottom = int(frameH * box[6])
            
            left = max(0, min(left, frameW - 1))
            top = max(0, min(top, frameH - 1))
            right = max(0, min(right, frameW - 1))
            bottom = max(0, min(bottom, frameH - 1))
            
            # Extract the mask for the object
            classMask = mask[classId]

            # Draw bounding box, colorize and show the mask on the image
            drawBox(frame, classId, score, left, top, right, bottom, classMask)

# Load names of classes
classesFile = "mscoco_labels.names";
classes = None
with open(classesFile, 'rt') as f:
   classes = f.read().rstrip('\n').split('\n')

# Give the textGraph and weight files for the model
textGraph = "./mask_rcnn_inception_v2_coco_2018_01_28.pbtxt";
modelWeights = "./mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb";

# Load the network
net = cv.dnn.readNetFromTensorflow(modelWeights, textGraph);

if args.device == "cpu":
    net.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)
    print("Using CPU device")
elif args.device == "gpu":
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    print("Using GPU device")

# Load the classes
colorsFile = "colors.txt";
with open(colorsFile, 'rt') as f:
    colorsStr = f.read().rstrip('\n').split('\n')
colors = [] #[0,0,0]
for i in range(len(colorsStr)):
    rgb = colorsStr[i].split(' ')
    color = np.array([float(rgb[0]), float(rgb[1]), float(rgb[2])])
    colors.append(color)

# def button_press_event(event):
#     global selectId
#     global Img_Mask
#     global Img_Mask_mod
#     global old_x
#     global old_y
#     global DrawEnable
#     # print("x y: " + str(event.xdata) + " " + str(event.ydata))
    
#     if len(Img) == 0:
#         return

#     numClasses = Img_Masks.shape[1]
#     numDetections = Img_boxes.shape[2]

#     frameH = Img.shape[0]
#     frameW = Img.shape[1]

#     for i in range(numDetections):
#         Img_Mask = np.zeros((frameH, frameW, 1), np.uint8)
#         Img_copy2 = Img.copy()
#         box = Img_boxes[0, 0, i]
#         mask = Img_Masks[i]
#         score = box[2]
#         if score > confThreshold:
#             classId = int(box[1])
                
#             # Extract the bounding box
#             left = int(frameW * box[3])
#             top = int(frameH * box[4])
#             right = int(frameW * box[5])
#             bottom = int(frameH * box[6])
                
#             left = max(0, min(left, frameW - 1))
#             top = max(0, min(top, frameH - 1))
#             right = max(0, min(right, frameW - 1))
#             bottom = max(0, min(bottom, frameH - 1))
                
#             # Extract the mask for the object
#             classMask = mask[classId]
#             classMask = cv.resize(classMask, (right - left + 1, bottom - top + 1))
#             mask = (classMask > maskThreshold)
#             roi = Img[top:bottom+1, left:right+1][mask]
#             Img_Mask[top:bottom+1, left:right+1][mask] = 255

#             if Img_Mask[int(event.ydata)][int(event.xdata)] > 0:
#                 selectId = i
#                 Img_Mask_mod = Img_Mask.copy()
#                 print("selected Id: " + str(selectId))
#                 Img_copy2[top:bottom+1, left:right+1][mask] = ([0.3*0, 0.3*255, 0.30] + 0.7 * roi).astype(np.uint8)
#                 Img_unit8 = Img_copy2.astype(np.uint8)

#                 ax1.imshow(Img_unit8)
#                 Canvas.draw() #Canvasへ描画
#                 break

def Quit():
    root.quit()
    root.destroy()

def PlotFile(canvas, ax, colors = "gray"):
    typ = [('JPG','*.jpg'),('PNG','*.png'),('BMP','*.bmp')] 
    dir = './'
    fle = filedialog.askopenfilename(filetypes = typ, initialdir = dir) 
    ax.cla() #前の描画データの削除
    global Img_Masks
    global Img_boxes
    global Img
    Img_Masks = []
    Img_boxes = []
    Img = []

    Img = cv.imread(fle)
    print("Img", Img.shape)

    Img_copy = Img.copy()
    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(Img, swapRB=True, crop=False)
    # Set the input to the network
    net.setInput(blob)
    # Run the forward pass to get output from the output layers
    Img_boxes, Img_Masks = net.forward(['detection_out_final', 'detection_masks'])
    # Extract the bounding box and mask for each of the detected objects
    postprocess01(Img_copy, Img_boxes, Img_Masks)
    # Put efficiency information.
    t, _ = net.getPerfProfile()
    label = 'Mask-RCNN on 2.5 GHz Intel Core i7 CPU, Inference time for a frame : %0.0f ms' % abs(t * 1000.0 / cv.getTickFrequency())
    cv.putText(Img_copy, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    Img_unit8 = Img_copy.astype(np.uint8)

    line = ax.imshow(Img_unit8)
    canvas.draw() #Canvasへ描画
    toolbar.update()

    numClasses = Img_Masks.shape[1]
    numDetections = Img_boxes.shape[2]

    frameH = Img.shape[0]
    frameW = Img.shape[1]

    for i in range(numDetections):  
        Img_copy2 = np.zeros((frameH, frameW, 3), np.uint8)
        box = Img_boxes[0, 0, i]
        mask = Img_Masks[i]
        score = box[2]
        if score > confThreshold:
            classId = int(box[1])
            
            # Extract the bounding box
            left = int(frameW * box[3])
            top = int(frameH * box[4])
            right = int(frameW * box[5])
            bottom = int(frameH * box[6])
            
            left = max(0, min(left, frameW - 1))
            top = max(0, min(top, frameH - 1))
            right = max(0, min(right, frameW - 1))
            bottom = max(0, min(bottom, frameH - 1))

            # Extract the mask for the object
            classMask = mask[classId]
            classMask = cv.resize(classMask, (right - left + 1, bottom - top + 1))
            mask = (classMask > maskThreshold)
            Img_copy2[top:bottom+1, left:right+1][mask] = ([255, 255, 255])

            frame_mask = Img_copy2.astype(np.uint8)
            frame_maskFile = "mask_" + str(i) + ".jpg"
            cv.imwrite(frame_maskFile, frame_mask)
            
if __name__ == "__main__":
    try:
        #Canvas.get_tk_widget().grid(row = 0, column = 0, rowspan = 10)
        Canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
        #UPDATEボタン
        ButtonWidth = 15
        UpdateButton = tkinter.Button(text="Load & GetMasks", width=ButtonWidth, command=partial(PlotFile, Canvas, ax1))#ボタンの生成
        UpdateButton.pack()
        #QUITボタン
        QuitButton = tkinter.Button(text = "QUIT", width = ButtonWidth, command = Quit) #QUITボタンオブジェクト生成
        QuitButton.pack()
        # tool bar
        toolbar = NavigationToolbar2Tk(Canvas, root)
        # Canvas.mpl_connect('button_press_event', button_press_event) 
        root.mainloop()
    except:
        import traceback
        traceback.print_exc()