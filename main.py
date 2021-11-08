#importing the required libraries
import numpy as np 
import cv2
import time
from keras.models import load_model

def detector(path_to_weights, path_to_cfg,path_to_sign_names,path_to_sign_classes,path_to_model,HEIGHT=32,WIDTH=32,detection_confidence=0.5,confidence_threshold = 0.5,
    font = cv2.FONT_HERSHEY_SIMPLEX):
    
    #creating the yolo (neural network) object
    net = cv2.dnn.readNet(path_to_cfg, path_to_weights)
    
    #reading classes from the signs_names file
    classes = []
    with open(path_to_sign_names, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    #getting all the layers' names from the yolo object
    layer_names = net.getLayerNames()

    #getting the output layer names which are present in the network 
    output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
    
    #declaring required variables
    colors = np.random.uniform(0,255,(len(classes), 3))
    start_time = time.time()
    frame_count = 0
    
    #loading the model for inference of the traffic signs(classification)
    classification_model = load_model(path_to_model)

    #creating a list for storing the different classes of our dataset
    classes_classification = []
    
    #reading different classes from the signs_names file
    with open(path_to_sign_classes, "r") as f:
        classes_classification = [line.strip() for line in f.readlines()]
    
    #creating an object to capture video from the camera
    video_capture = cv2.VideoCapture(0)

    while True:
        #reading the frame
        res,img=video_capture.read()

        #if the frame is not read, then break
        if not res:
            break
        
        #incrementing the frame count in each iteration to calculate the FPS
        frame_count +=1
        
        #getting the height, width and channels of the frame
        height, width, channels = img.shape

        #preprocessing the image
        blobIm = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        
        #setting the blobbed image to the neural network for detection of traffic signs
        net.setInput(blobIm)

        #getting the detected signs in the frame
        outs = net.forward(output_layers)

        #making list variables to store ids,confidences,boxes of the detected boxes
        class_ids = []  #class ids for storing the ids 
        confidences = [] #confidences for storing the confidences of each predicted box class
        boxes = [] #boxes for storing the coordinates of each predicted box

        for out in outs:    #iterating over multiple outputs
            for detection in out:   #iterating over all the detected boxes in each output layer
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > confidence_threshold:#traffic sign detected
                    
                    #calculating center of the bounding box
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    
                    #coordinates of the bounding box
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    #appending to list of boxes,confidences,class_ids
                    boxes.append([x, y, w, h]) #x - starting x coordinate,y-starting y coordinate, w-width, h-height
                    confidences.append(float(confidence))   #confidences for each box
                    class_ids.append(class_id) #class_ids for each box
        
        #suppressing the redundant/multiple boxes using non max suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        #displaying the boxes
        for i in range(len(boxes)): #iterating over all the boxes to classify the traffic sign present in them
            
            if i in indexes:    #checking if the box's index is present even after non max suppression
                
                x, y, w, h = boxes[i]
                
                #getting the category of class to which the detected traffic sign belongs to
                #label = str(classes[class_ids[i]]) + "=" + str(round(confidences[i]*100, 2)) + "%"
                
                #drawing the bounding box at the predicted coordinates
                cv2.rectangle(img, (x, y), (x + w, y + h), colors[class_ids[i]], 2)
                
                #cropping the image
                crop_img = img[y:y+h, x:x+w]
                
                if len(crop_img)>0:#if the image is not empty
                    #resizing the image to the required width and height as per the arguments passed to our classification model
                    crop_img = cv2.resize(crop_img, (WIDTH, HEIGHT))
                    crop_img =  crop_img.reshape(-1, WIDTH,HEIGHT,3)
                    
                    #getting the predicted class's index of the traffic sign to identify it's class
                    prediction = np.argmax(classification_model.predict(crop_img))
                    
                    #getting the class name of the bounding box 
                    label = str(classes_classification[prediction])
                    
                    #displaying the label on the output video
                    cv2.putText(img, label, (x, y), font, 1, colors[class_ids[i]], 3)
        
        #calculating current time
        final_time=time.time()
        spent_time = final_time-start_time
        
        #calculating the fps
        fps = frame_count/spent_time

        #displaying the fps on the output video
        cv2.putText(img,'FPS: '+str(round(fps, 2)),(20,40),font,1,(0,0,255),3)
        
        #printing fps to console for debugging purposes
        print("fps: ", str(round(fps, 2)))
        
        #displaying the image
        cv2.imshow("Image", img)
        
        #checking for key press, if q is pressed then break from the loop and stop the execution
        if cv2.waitKey(1) & 0xFF == ord ('q'):
            break
    
    
    #releasing the camera and destroying the windows
    cv2.destroyAllWindows()

#calling the main function if the file is run as a script
if __name__=='__main__':
    #change this path according to your system
    default_path='C:/Users/venka/OneDrive/Desktop/traffic-sign-detection/Input/'
    path_to_weights = default_path+'yolov3_training.weights'
    path_to_cfg = default_path+'yolov3_training.cfg'
    path_to_sign_names=default_path+'sign_names.txt'
    path_to_sign_classes=default_path+'signs_classes.txt'
    path_to_model1=default_path+'traffic.h5'
    detector(path_to_weights, path_to_cfg, path_to_sign_names, path_to_sign_classes, path_to_model1)
#end of the code