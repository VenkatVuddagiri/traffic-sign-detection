//import opencv and tensorflow
#include "opencv2/opencv.hpp"
#include<iostream>
#include<string>
#include<fstream>
#include<string>
using namespace std;
using namespace cv;

vector<string> readClassNames(string path){
    //read class names from the path into a vector of strings
    ifstream ifs(path);
    if(!ifs.is_open()){
        cout<<"error opening file"<<endl;
        exit(1);
    }
    string line;
    vector<string> classNames;
    while(getline(ifs,line)){
        classNames.push_back(line);
    }
    return classNames;
}

void detector(string path_to_weights,string path_to_cfg,string path_to_sign_names,string path_to_sign_classes,string path_to_model){
    //creating yolo neural network from opencv
    cv::dnn::Net net = cv::dnn::readNetFromDarknet(path_to_cfg, path_to_weights);
    //reading classes from sign_names file
    vector<string>classes=readClassNames(path_to_sign_names);
    //getting the layer names of the neural network
    vector<string>layer_names=net.getLayerNames();
    //getting the names of the output layers of the neural network
    vector<string>output_layers=net.getUnconnectedOutLayersNames();
    //capturing video from camera
    VideoCapture cap(0);
    //checking if the video is opened
    if(!cap.isOpened()){
        cout<<"Error opening video stream or file"<<endl;
        return;
    }
    //creating a window to display the video
    namedWindow("Video",WINDOW_NORMAL);

    //reading and classifying the frames
    while(1){
        //reading the frame
        Mat frame;
        cap>>frame;
        //checking if the frame is empty
        if(frame.empty()){
            cout<<"Error reading video stream"<<endl;
            break;
        }
        //resizing the frame
        resize(frame,frame,Size(640,480));
        //converting the frame to grayscale
        cvtColor(frame,frame,COLOR_BGR2GRAY);
        //getting the output at the output layeyrs of the neural network
        vector<Mat>outputs;
        net.setInput(cv::dnn::blobFromImage(frame,1/255.0,Size(416,416,3),Scalar(0,0,0),true,false));
        net.forward(outputs,output_layers);
        //getting the number of detected objects    

        //checking if the predictions are empty
        if(outputs.size()==0){
            cout<<"No object detected"<<endl;
            break;
        }
        //creating a copy of the frame
        Mat frame_copy=frame.clone();
        //looping over the predictions
        for(int i=0;i<outputs.size();i++){
            //getting the prediction
            vector<float>prediction=outputs[i];
            //getting the class id
            int class_id=prediction[0];
            //getting the confidence
            float confidence=prediction[2];
            //getting the class name
            string class_name=classes[class_id];
            //getting the bounding box coordinates
            int left=prediction[3]*frame.cols;
            int top=prediction[4]*frame.rows;
            int right=prediction[5]*frame.cols;
            int bottom=prediction[6]*frame.rows;
            //checking if the confidence is greater than 0.5
            if(confidence>0.5){
                //drawing the bounding box
                rectangle(frame_copy,Point(left,top),Point(right,bottom),Scalar(0,255,0),2);
                //writing the class name
                putText(frame_copy,class_name,Point(left,top),FONT_HERSHEY_SIMPLEX,1,Scalar(0,255,0),2);
            }
        }
        //displaying the frame
        imshow("Video",frame_copy);

        //checking for key press, if q is pressed then break from the loop and stop the execution
        if(waitKey(1)=='q'){
            break;
        }
    }

    //releasing the video
    cap.release();
    //destroying all the windows
    destroyAllWindows();
}

//main function
int main(){
    string default_path="C:/Users/venka/OneDrive/Desktop/traffic-sign-detection/Input/";
    string path_to_weights = default_path+"yolov3_training.weights";
    string path_to_cfg = default_path+"yolov3_training.cfg";
    string path_to_sign_names=default_path+"sign_names.txt";
    string path_to_sign_classes=default_path+"signs_classes.txt";
    string path_to_model1=default_path+"traffic.h5";
    detector(path_to_weights, path_to_cfg, path_to_sign_names, path_to_sign_classes, path_to_model1);
    return 0;
}