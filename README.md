https://drive.google.com/file/d/1Y4rbMdUENOPFKVAVYMAffNFZla5PSys5/view?usp=drive_link


Abstract-Recognizing the importance of human computing interaction in particular vision-based gesture and object recognition is to control computers and other devices with gestures rather than pointing and clicking a mouse or touching a display directly. 
In this project, we use the detection of hand gestures which are portrayed by a webcam linked with a computer. OpenCV is an open-source library that mainly focuses on real-time image processing. Here we capture hand movements by applying a machine learning algorithm that is pre-trained with datasets to detect hand gestures. By using the PYAUTOGUI python library we perform mouse operations accordingly.

I. Introduction
Recent developments in technologies have provided a value added services to the users. In everyday life we often use gestures for mean of communication. Using such gestures we can communicate with computers without contact. The pandemic has made us more aware of the physical distancing and how hypersensitive we have to be to everything we touch. Public services like ATM’s often visited by so many people and there is a risk of spreading virus through the process of using these machines (touching keypads). Touch-less interfaces provide a means of communication with machines without physical contact. The task of touchless interfaces is to communicate using gestures, hand gesture recognition is one elemental problem in computer vision. Recent advances in technologies provided ways for processing tasks like hand detection, hand recognition, and hand tracking. Gesture recognition can be done with techniques from computer vision and image processing.  The significance of gesture recognition lies in constructing effective human-machine interaction. Its applications are widespread in various industries like robotics, virtual reality, and biometrics. In gesture recognition technology, a camera reads the movements of the human body and communicates the data to a computer that processes the data, and uses the gestures as input to control devices or applications. For example, in our application, a person using a gesture of finger count 1 in front of a camera can trigger mouse pointer control.

Hand gestures are an aspect of body language that can be conveyed through the finger position and the shape constructed by the hand. Hand gestures can be categorized into static and dynamic. Static gesture refers to those that only require the processing of a single image at the input of the classifier, whereas dynamic gesture requires the processing of image sequences and more complex gesture recognition approaches. we can find several recognition methodologies based on supervised and unsupervised learning. We can cite some examples, such as neural networks, convolutional neural networks, support vector machines (SVM), nearest neighbors, graphs, distributed locally linear embeddings, and others.
Convolutional Neural Networks (CNNs), a branch of deep learning, have an impressive record for image analysis and interpretation applications, including medical imaging. Presently, CNNs are used to successfully tackle highly complex image recognition tasks with many object classes.
This project emphasizes the recognition of static hand gestures by building a model using CNN that can analyze large amounts of image data, recognize static hand gestures, and perform corresponding actions assigned to that gesture. The other objective is to use these gesture control actions for contactless interaction with an ATM interface.


II. Related Works
Computers are an integral component of our daily lives and are employed in a variety of fields. Traditional input devices such as the mouse and keyboard facilitate human-computer interaction. Hand gestures can be an effective means of human-computer interaction and can make it easier. Human hand gestures can be characterized as a set of permutations produced by hand and arm movements [1]. Hand gesture recognition has been the subject of numerous studies, and some noteworthy research in this area is discussed. 
	The study [2] developed a hand gesture recognition system based on the form-fitting methodology and an artificial neural network (ANN). After filtering, a color segmentation approach on the YbrCr color space was employed to recognize the hand in this system. The hand morphology then approached the shape of the hand. An ANN was used to extract the shape of hands and finger orientation information. Using this strategy, they were able to obtain an accuracy of 94.05%
	A statistical technique for detecting gestures based on haar-like traits has been suggested [4]. The AdaBoost technique was utilized to learn the model in this system. Manual feature extraction, on the other hand, has several disadvantages. The extraction procedure is time-consuming, and it is unlikely that all available features will be extracted. 
An approach for detecting gestures using CNN has been proposed [6], which is robust to five invariants: scale, rotation, translation, illumination, noise, and background. The dataset was Peruvian Sign Language (LSP). On the LSP Dataset, they were 96.20 percent accurate.
An algorithm to recognize hand gestures using a 3D CNN was proposed [5]. In this system, the basis of the recognition was the challenging depth and intensity of the images. They also used a data augmentation technique and achieved an accuracy of about 77.5% on the VIVA challenge dataset [5].
Chai et al. [3] offer a hand gesture application in a 3D depth data processing approach for gallery browsing. In the gesture framework created, it integrates global structure information with local texture variation. In their work, Pavlovic et al. [7] concluded that in order to create an effective human-computer interface, user motions must be rationally explainable. One of the key issues that have emerged throughout is the time complexity and robustness that is involved with gesture recognition analysis and evaluation.
Paper [8] posed a technical and conceptual foundation for the execution of the work. For classification, they employed the “bag-of-feature (BOF)” method and a nonlinear “support vector machine (SVM).” Despite its limitations, such as solely taking into account unordered attributes, it became quite popular for object classification and obtained an accuracy of 97%.



III. Methodology
This section will provide the details of the dataset, CNN, and ATM GUI used in the project. The flowchart of the methodology is shown in figure 1. The approach included collection of required datasets, pre-processing, configuring the CNN model, capturing frames using open-cv for prediction and integrating with ATM GUI.
A. Dataset 
We used a dataset named “Fingers” (reference) which was publicly available, the dataset consist of 6 static gestures (fingers count: 0,1,2,3,4,5) to recognize. Each class had 3000 images for training and 1000 images for testing. So total images of 24000 and all images are 128 by 128 pixels. Images are centered by the center of mass and noise pattern on the background.
 
B. Pre-processing
            We categorized the dataset into 6 respective classes and divided them into training and testing sets with labels and features. We applied a minimal pre-processing over the dataset to reduce the computational complications. The images were in grayscale, we applied GaussianBlur to reduce the noise from the image, threshold to completely convert the image pixel to max over a threshold. The images are resized to 128*128 for feeding to CNN.
C. CNN configuration
            The CNN model used in this project composed of 6 conv2d layers,9 batch-normalization layers, three fully connected layer and output layer. Three dropout performance used in the network to prevent over-fitting.
            The first layer has 64 different filters with the kernel size of 3*3, ReLU activation function was used to introduce non-linearity. The ReLU activation function perform better than other function such as sigmoid or tansh. We have to specify the input size for first conv2d layer. The input shape is 128*128*1, which represents gray scale image of 128*128 size. The Stride is set to default (1*1). The feature maps produce by this layer are passed to next layer.
            Then the CNN has a batch-normalization is used to apply a transformation that maintains the mean activation close to 0 and the activation standard deviation close to 1. This stabilizes and accelerate the learning process. Batch-normalization attempts to resolve an issue in neural network called internal covariate shift.
            The second and third layers are another convolution layers with 64 different filters with the kernel size of 3*3, default stride for second layer and 2*2 stride for third layer. ReLU activation function is used in both layers. Both layers are followed by another batch-normalization. In third layer first dropout was added which randomly discards 20% neurons to prevent the model from over-fitting. The output from this layer is passed to next layer. 
            The four, five and six layers are another convolution layers with 128 different filters with the kernel size of 3*3(four, five) and 5*5(sixth), default stride for fourth, fivth layers and 2*2 stride for sixth layer. ReLU activation function is used in all three layers. Three layers are followed by another batch-normalization. In sixth layer second dropout was added which randomly discards 30% neurons. Output from this layer is passed to flatten layer. The input applied to flatten layer is converted to vector, it will allow the fully connected layers to process the data achieved till now.
            The next three layers are fully connected layers with 256,128,128 nodes respectively to receive the vector produced by the previous layer. All three layers uses ReLU activation function. Third dropout layer is introduced followed by these layers to exclude 40% of the neurons to prevent over-fitting.
            The output layer has 6 nodes corresponding to each class of the hand gesture, it uses softmax activation function which outputs a probabilistic value for each class. Stochastic Gradient Descent (SGD) [18] function with a learning rate 0.01 is used while compiling the model. Categorical cross-entropy function was used to evaluate the loss. Finally, the metrics of accuracy were specified to keep track on the evaluation process. Summary of the CNN configuration is shown on Table I.
 
D. Capturing frames using open-cv for prediction            
            The first step was to capture the frames using videoCapture function through web camera. Then extract desired region or region of interest from the frame. The frame undergoes a series of operations before passing to input layer of CNN. The hand is segmented from the frame using background subtraction method, it is commonly used technique for generating a foreground mask. The obtained image than converted into grayscale and applied with gaussian blur function to reduce the background noise. Threshold function is used to completely convert the image pixel to max over a threshold limit. The image is resized to 128*128 and passed to input layer of CNN using predict function of saved model. Before passing the input to predict function we have to load the saved model. 
    
E. ATM graphical interface and Integration
            We developed an ATM graphical user interface using Tkinter a standard GUI library for python. This interface is used to demonstrate gesture based human interaction. The ATM GUI is provided with an authentication page where user will provide there password. A virtual keypad was provided in auth page so that the user can enter their password by controlling mouse using gestures. There is a menu page where the user is redirected after their authentication, it has different options to select like withdraw, deposit, balance and more. Each option has its own individual page for their respective operations. 
            The gesture recognition code is embedded in app. The Original video streaming frame is provided in the app for hand position reference. Output frame is also provided in app to display predicted output
F. System implementation     

            To implement the system, python was used as the programming language and a python IDE jupyter notebook was used to write and run the code. The library open-cv was used to capture the frames from the web camera and image processing. PIL and Imutils were used for image pre-processing. Keras was used to build the CNN classifier. Tkinter and pyautogui were used for GUI development and mouse controlling respectively. Numpy was used for array operations. Matplotlib was used to visualize model accuracy and loss values.
 
            In Training phase, the model was trained using the base dataset achieved after pre-processing. Data augmentation techniques such as  cropping, padding and horizontal flipping are used to increase the data used for training the model. It also acts as regularizer and helps reduce overfitting.



Experimental results
This chapter describes the results achieved in the project. We tested our proposed system for categorizing gesture into one of six categories and perform corresponding action. The learning rate and the number of epochs are tuned to train the model.  
The model is trained for 10  epochs with a batch size of 32.The results obtained after training of CNN classifier shows that the model achieved 98.57% accuracy with a minimal loss of 1.32%.The performance of the model is shown in Table.
Epoch	Accuracy	loss	Value accuracy	Value
Loss
1	90.18	3.12	5.54	82.95
3	96.43	2.82	2.97	97.63
5	97.62	2.21	2.13	97.73
7	97.72	1.56	1.34	97.89
9	98.34	1.45	1.27	98.59
10	98.57	1.32	1.16	98.78

The relation between Train loss, validation loss, accuracy, validation accuracy and epoch is shown in the figure. The proposed model yields a loss of 1.16%. 
 
The predicted output results are shown in the following figures.
 
Figure: displaying gesture indicating zero count.
 
Figure: displaying gesture indicating two count.
 
Figure: displaying gesture indicating four count.
	The gesture indicating count 2 performs click action, while gesture indicating 1,3,4,5 perform movement of mouse cursor down, up, left, right from current cursor position respectively. 


Conclusion and future scope
Conclusion:
	Gesture provides a mean of communication involving physical movement of our body with the intent of conveying message or interacting with the environment. The significance of gesture recognition lies in constructing effective human machine interaction. The application of gesture recognition lies in wide range of industries or technologies like robotics, gaming, virtual reality, medical profession, biometrics etc. Adding support for various types of gestures to electronic devices enables to operate these devices without touching, which is much more intuitive and effortless when compared to touching a screen, manipulating a mouse, keyboard, pressing a switch.	
	Gesture controls will notably contribution to easing interaction with devices, reducing the need for keys, buttons, mouse. When combined with other advanced user interface technologies voice commands and face recognition can create a richer user experience.
Future Scope:
The gesture recognition is most significant aspect/ technology within the worldwide. Interactions between humans and IT equipment is everywhere. So, to exhibit the interaction between humans, devices, machines in a very virtual platform there’s an automatic gesture recognition. We will use a greater number of gestures in a very particular frame of your time. Gesture is nothing but the recognition of the movement of the body or an element of the body.
Here, during this project we’ve only used static gestures, which means a particular hand configuration and a pose, represented by a single image. So, we can also improve using dynamic recognition which involves more complex gestures. Dynamic gestures provide an effective way of interaction. Such type of dynamic gesture recognition systems is used mostly in medical fields, food industries, video games, sign language interpretation, and so on.
Today, gesture recognition is a real thing. Many companies like Intel and Microsoft have already created use cases for this technology. Gesture recognition will change our relationship with the technical devices. In future the gesture recognition cab be placed in hospitals with advanced robotic systems and can be placed in homes. It will be also employed in automation systems like in homes, vehicles, to greatly increase usability and reduce the resources to create input systems like remote controls, car entertainment systems. It can also used for an earlier life for the disabled persons. We train them using datasets, without touching we use them through gestures. If suppose we would like to swipe the channel without using the remote-control system, we use hand gestures, if we just use our hand by moving, TV recognizes and the channel changes. In such how we train the datasets. It can also be used to switch on and off the fan and so on. Thus, the future applications of gesture recognition can be used and make them work accordingly.


REFERENCES
[1]Brutzer, S.; Hoferlin, B.; Heidemann, G.         Evaluation of Background Subtraction Techniques for Video Surveillance. In Proceedings of the 24th IEEE Conference on Computer Vision and Pattern Recognition,Colorado Springs, CO, USA, 20–25 June 2011; pp. 1937–1944.
Shaikh, S.; Saeed, K.; Chaki, N. Moving Object Detection Approaches, Challenges and Object Tracking. In Moving Object Detection Using Background Subtraction, 1st ed.; Springer International Publishing: NewYork, NY, USA, 2014; pp. 5–14.
S. M. M. Roomi, R. J. Priya, and H. Jayalakshmi 2010 Hand Gesture Recognition for Human-Computer Interaction (J. Comput. Science vol. 6) no. 9 pp. 1002–1007.
S. N. Karishma and V. Lathasree 2014 Fusion of Skin Color Detection and Background Subtraction for Hand Gesture Segmentation (International Journal of Engineering Research and Technology) vol. 3 no 1 pp 13–18. 
N. Otsu 1979 A Threshold Selection Method from Gray-Level Histograms (IEEE Trans. Syst. Man. Cybern.) vol. 9 no. 1 pp 62–66 
S. van der Walt, S. C. Colbert, and G. Varoquaux 2011 The NumPy array: a structure for efficient numerical computation (Comput. Sci. Eng.) vol. 13 no. 2 pp 22–30 
Travis E. Oliphant et al 2017 NumPy Developers numpy 1.13.0rc2. (Online) Available: https://pypi.python.org/pypi/numpy (Accessed: 13-Oct-2017).
M. Nahar and M. S. Ali 2014 An Improved Approach for Digital Image Edge Detection (Int. J. Recent Dev. Eng. Technology) vol. 2 no. 3 
S. Lenman, L. Bretzner and B. Thuresson. "Computer Vision Based Hand Gesture Inter-faces for Human-Computer Interaction", Technical Report CID-172, Center for User Oriented IT Design, pp.3--4, 2002.
V. I. Pavlovic, R. Sharma, and T. S. Huang (1997), “Visual interpretation of hand gestures for human computer interaction.” IEEE Trans. Pattern Analysis and Machine Intelligence 19(7) : 677–695.
Orazio, Marani, Reno and Cicirelli (2016), “Recent trends in gesture recognition: how depth data has improved classical approaches.” Image and Vision Computing 52 : 56–72.
Neto G.M.R., Junior G.B., de Almeida J.D.S., de Paiva A.C. (2018), “Sign Language Recognition Based on 3D Convolutional Neural Networks.” In: Campilho A., Karray F., ter Haar Romeny B. (eds) Image Analysis and Recognition. ICIAR Lecture Notes in Computer Science 10882.
Mohanty A., Rambhatla S.S., Sahay R.R. (2017), “Deep Gesture: Static Hand Gesture Recognition Using CNN.” In: Raman B., Kumar S., Roy P., Sen D. (eds) Proceedings of International Conference on Computer Vision and Image Processing. Advances in Intelligent Systems and Computing, Springer, Singapore 460.
Lecun, Y., Bengio, Y. and Lhinton G. (2015). “Deep learning.” Nature 521 : 436–444

