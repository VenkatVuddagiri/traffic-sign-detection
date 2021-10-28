#importing the required libraries
import numpy as np 
import cv2
import time
from keras.models import load_model
import os

#setting the width and height of the image
HEIGHT = 32
WIDTH = 32

def main():
    #setting the path to the weights and config files
    path_to_weights = 'Input/yolov3_ts_train_5000.weights'
    path_to_cfg = 'Input/yolov3_ts_train.cfg'
    
    #creating the yolo object
    net = cv2.dnn.readNetFromDarknet(path_to_cfg, path_to_weights)
    
    #reading classes from the signs_names file
    classes = []
    with open("Input/sign_names.txt", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    #getting layers names from the yolo object
    layer_names = net.getLayerNames()

    #getting the output layer names
    output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
    
    #setting parameters
    colors = np.random.uniform(0,255,(len(classes), 3))
    check_time = True
    confidence_threshold = 0.5
    font = cv2.FONT_HERSHEY_SIMPLEX
    start_time = time.time()
    frame_count = 0
    detection_confidence = 0.5
    
    #loading the model
    classification_model = load_model('Input/traffic.h5')
    classes_classification = []
    #reading different classes from the signs_names file
    with open("Input/sign_classes.txt", "r") as f:
        classes_classification = [line.strip() for line in f.readlines()]
    
    #capturing the video
    video_capture = cv2.VideoCapture(0)

    while True:
        #reading the frame
        res,img=video_capture.read()
        if not res:#if the frame is not read correctly then break
            break
        frame_count +=1#incrementing the frame count
        height, width, channels = img.shape#getting the height, width and channels of the frame

        #preprocessing the image
        blobIm = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        
        #setting the blobIm to the darknet network
        net.setInput(blobIm)

        #getting the detections
        outs = net.forward(output_layers)

        #making list variables to store ids,confidences,boxes
        class_ids = []
        confidences = []
        boxes = []

        for out in outs:#iterating over multiple outputs
            for detection in out:#iterating over detections in each output
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > confidence_threshold:#object detected
                    #calculating center of the bounding box
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    #coordinates of the bounding box
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    #appending to list of boxes,confidences,class_ids
                    boxes.append([x, y, w, h]) #x - starting x coordinate,y-starting y coordinate
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        #suppressing the redundant/multiple boxes 
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        #displaying the boxes
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                #labeling the boxes
                label = str(classes[class_ids[i]]) + "=" + str(round(confidences[i]*100, 2)) + "%"
                #drawing the bounding box
                cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 2)
                #cropping the image
                crop_img = img[y:y+h, x:x+w]
                if len(crop_img)>0:#if the image is not empty
                    #resizing the image
                    crop_img = cv2.resize(crop_img, (WIDTH, HEIGHT))
                    crop_img =  crop_img.reshape(-1, WIDTH,HEIGHT,3)
                    #predicting the image
                    prediction = np.argmax(classification_model.predict(crop_img))
                    #predicting the class of the bounding box
                    label = str(classes_classification[prediction])
                    #displaying the label
                    cv2.putText(img, label, (x, y), font, 0.5, (255,0,0), 2)
        #calculating current time
        final_time=time.time()
        spent_time = final_time-start_time
        #calculating the fps
        fps = frame_count/spent_time
        cv2.putText(img,'FPS: '+str(fps),(20,40),font,1,(255,255,0),5)
        print("fps: ", str(round(fps, 2)))
        #displaying the image
        cv2.imshow("Image", img)
        #checking for key press and breaking the loop
        if cv2.waitKey(1) & 0xFF == ord ('q'):
            break
    #releasing the camera
    cv2.destroyAllWindows()

#calling the main function if the file is run as a script
if __name__=='__main__':
    main()
#end of the code