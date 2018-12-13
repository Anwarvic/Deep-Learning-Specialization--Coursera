#Note

In the 'Autonomous driving application - Car detection' notebook, we are using a file called 'yolo.h5'
located at 'model_data' directory. This file can be obtained following these steps:

- download the yolo weights via this (link)[http://pjreddie.com/media/files/yolo.weights]
- download the configuration file via this (link)[https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolo.cfg]
- put these two files inside ./yad2k/models direcotry
- run the following command `./yad2k.py yolo.cfg yolo.weights ../../model_data/yolo.h5`


Or simply, download it from [here](http://www.mediafire.com/file/39rgr73zztk9hue/yolo.h5/file)