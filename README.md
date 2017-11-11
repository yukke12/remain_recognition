# About this repository
To make a system which detect a mount of remaining by foods.

# Enviroument
Ubuntu:16.04
CUDA:8.0
CUDNN:6
Python:Python 3.5.2 :: Anaconda 4.2.0 (64-bit)

# Others
## how to use camera module
### check the detect for camera
```
vcgencmd get_camera
```

### take a photo sample command.
```
raspistill -o 100_image-%004d.jpg -tl 3000 -t 60000 -w 640 -h 480
```

-o : image_name  
-tl : every mili second(Above command mean every 3 seconds take a photo)  
-t : all mili second(Above command mean all time is 60 seconds=1 minitus)  
-w : weigh  
-h : hight

## About raspberry pi avairable links
[camera module](https://www.rs-online.com/designspark/raspberry-pi-camera)  
[camera module2](http://nagashy.hatenablog.com/entry/2017/01/12/093116)  
[connect use mac and raspi only](https://qiita.com/mascii/items/7d955395158d4231aef6)  
[ssh connect using free LAN](http://darmus.net/raspberry-pi-ssh-mac-terminal/)  
[allow to connect from camera module to raspi](http://tomosoft.jp/design/?p=8911)  
