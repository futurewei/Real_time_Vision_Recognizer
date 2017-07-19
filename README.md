# Vision_recognizer


In this project, I try to implement a real time vision recognizer. It's been a long time since I wanted to integrate deep nerual network with openCV. But for some reason, mac is unable to use many features such as cv2.imshow of openCV.
But I just finally got it work by creating a virtual space on my laptop.


The first task for me is to identify a person, and myself will be the perfect candidate to be identified. Since I don't have a big dataset of images of myself, I just did a image similarity check using my dot product. Also, to extract my face as region of interest and save it to be analyzed, I used the face and eyes cascaded from openCV. If similarity goes above 75%, that means it identifies the personas me, and output the text.

![accuracy](https://user-images.githubusercontent.com/13871858/28259726-82500084-6a8c-11e7-8567-acd8eb909435.jpg)


It would be very lame to only use dot product to do image recognition. Thus, I wrote a Deep convolutional Neural network using tensorflow to identify objects. It's a binary network, which classifies the image as books and others. Ideally, it should be trained for 10 thousands many times, but considering my laptop is Mac air and also the timing cost, I decided to only run it 500 times, training about 8000 images. 

The book dataset I used for Deep convolutional Neural network is from ImageNet, as shown here:
![23 pic](https://user-images.githubusercontent.com/13871858/28357338-eb1eee42-6c1f-11e7-9c4c-39cffd18bebf.jpg)

But then came the hardest part! I had no idea how to choose my intereted region because book cascade is not implmented in openCV. I tried to divide the screen into different regions and feed into Neural Network, but since there's no any callback function in python, the code either didn't get executed or just completely stopped the video flow.
![2 pic](https://user-images.githubusercontent.com/13871858/28358328-552b4512-6c23-11e7-9599-7af7dca326ef.jpg)


I've also tried the contours and feature matching in openCV library. But it turns out to be very slow and ineffective. At the end, I realized only opencv cascade can achieve the effect of detectmultiscale by neighbors. So I just decided to make custom opencv cascade. After a few attempts, I really felt like openCV is user unfriendly to Mac users, so many command line tools can't get executed. Thus, I went ahead and opened a Linux cloud server on Digital Ocean. And then I uploaded my datatset using FileZilla, port 22, to my Digital Ocean server and start training. I trained 10 stages, and got my cascade.xml back and use it to identify and locate my book.

Even though cascade seem to do the same trick as my convolutional network, my main goal in this project is to practice and integrate my Neural network with openCV. So I still applied the Neural network to the region selected by cascade. At least, Neural network is able to give us the accuracy of the image.

There's a lot to be improved, but more like the number of training epoch, number of classifications. Thus, I'm just leaving it like this, but perhaps optimize it when I have free time.
(final demo accuracy was blocked by my terminal...but accuracy was shown on the screen of terminal ^ ^)
![4 pic](https://user-images.githubusercontent.com/13871858/28357341-eec8a8a8-6c1f-11e7-8b03-d641b43ff5d7.jpg)
