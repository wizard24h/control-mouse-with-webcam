# hand-gesturing-system-using-CNN-and-image-processing
### mouse + multimedia controlling system using hand gesturing with convolution neural network and image processing 

AIU
![Arabic International University](https://ibb.co/j8ygsxc)

## required depedencies: 
* python 3.7
* numpy
* cv2
* glob (file system manager)
* pynput (python input library mouse + keyboard)
* os (python system library for invoking system commands)
* keras (CNN model creation library)
* tensorflow 
* tkinter (python GUI library)
* PIL (Python Image Library)
* flask (python web framework for CAPTCHA system)

## System workflow
![my algorithm](https://ibb.co/x1mVC9j)

explanation:
1. start GUI.py using python 3.7 environment
2. the program checks if ./palms directory has previously saved palms or not
3. if empty then create new palm by placing your first over the green rectangle
![palm color saving](https://ibb.co/yfwmTNd)
4. make sure your first covers all the gaps in the green rectangle
5. press z once you done
6. system GUI will startup showing up the main tab (shortcuts tab)
![shortcuts tab](https://ibb.co/s34XXTY)
7. here you can assign simple gestures to cmd commands such as start program or shutdown computer
![Settings tab](https://ibb.co/BsHfVWh)
8. the next tab is settings where you can setup your personal preferences such as sensetivity and accuracy
![Palms Tab](https://ibb.co/XJtZQzD)
9. the last tab is used to create different set of palms (gloves) for different environments where lightining and background changes

### important directories and files
* ./dataset directory where we hold our generated dataset
* ./Images directory for images reference in GUI
* ./palms directory to save our different palms (gloves)
* CAPTCHA.py web server to be used as CAPTCHA system (ask 18-14 and user has to rise four fingers with his palm)
* classifier.py used to classify our palm to recognize the desired command
* dataset_creator.py used to generate our dataset (change file name in 39 and 42) then start the script to capture your palm frame by frame and save each to the dataset file
* DMImage.py simple library to process the image
* GUI.py our GUI window used to call reGesture.py (our main script)
* PalmLib.py our own implementation to detect and track user palm
* reGesture.py (our main script)
* windowsController.py use pynput to make simple system calls 


##  CNN model + model evaluation
![training dataset generator](https://ibb.co/6ZfwYDm)
![CNN model description](https://ibb.co/JzYvLP3)
![Algorithm evaluation](https://ibb.co/jVmLszM)


### my email wizard.24h@gmail.com
please contact me for any matter 

### my paypal mhamad982@gmail.com
please send donations in case you like my project 


best regards. 

