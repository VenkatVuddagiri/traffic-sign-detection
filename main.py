import numpy as np 
import cv2
import time
from keras.models import load_model
import os

HEIGHT = 32
WIDTH = 32


def main():
    path_to_weights = 'Input/yolo_training.weights'
    path_to_cfg = 'Input/yolov3_ts_train_5000.cfg'
    
    #loading the model into the memory 
    net = cv2.dnn.readNet(path_to_cfg, path_to_weights)
    
    #reading classes from the signs_names file
    classes = []
    with open("Input/signs_names.txt", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    #setting parameters
    colors = np.random.uniform(0, 255,(len(classes), 3))
    check_time = True
    confidence_threshold = 0.5
    font = cv2.FONT_HERSHEY_SIMPLEX
    start_time = time.time()
    frame_count = 0
    detection_confidence = 0.5
    cap = cv2.VideoCapture(0)
    
    #loading the model
    classification_model = load_model('traffic.h5')
    classes_classification = []
    with open("Input/signs_classes.txt", "r") as f:
        classes_classification = [line.strip() for line in f.readlines()]

    print('reached here')
    video_capture = cv2.VideoCapture(0)

    while True:
        res,img=video_capture.read()
        if(res):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_count +=1
        height, width, channels = img.shape
        window_width = width

        #preprocessing the image
        blobIm = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        
        
        net.setInput(blobIm)
        outs = net.forward(output_layers)

        #making list variables to store ids,confidences,boxes
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:#iterating over multiple outputs
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h]) #x - starting x coordinate,y-starting y coordinate
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        #suppressing the redundant/multiple boxes 
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        #displaying the boxes
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]]) + "=" + str(round(confidences[i]*100, 2)) + "%"
                cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 2)
                crop_img = img[y:y+h, x:x+w]
                if len(crop_img)>0:
                    crop_img = cv2.resize(crop_img, (WIDTH, HEIGHT))
                    crop_img =  crop_img.reshape(-1, WIDTH,HEIGHT,3)
                    prediction = np.argmax(classification_model.predict(crop_img))
                    label = str(classes_classification[prediction])
                    cv2.putText(img, label, (x, y), font, 0.5, (255,0,0), 2)

        
        #displaying image
        final_time=time.time()
        spent_time = final_time-start_time
        fps = frame_count/spent_time
        cv2.putText(img,'FPS: '+str(fps),(20,40),font,1,(255,255,0),5)
        print("fps: ", str(round(fps, 2)))
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord ('q'):
            break
    cv2.destroyAllWindows()


if __name__=='__main__':
    main()