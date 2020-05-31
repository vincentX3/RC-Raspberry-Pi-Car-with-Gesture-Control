# RC-Raspberry-Pi-Car-with-Gesture-Control
Have fun! Using gesture to control your Remote Control Raspberry Pi Car!

## What we done?
Using gesture to control a Raspberry car.

Technically, 

- we trained a neural network model for gesture recognition,
- Using Raspberry Pi with camera to provide video stream from car's view.
- Send commands to our toy car, which is controlled by ESP8266.

All the components are communicate under a local network.

## Hardware

- Raspberry Pi 4 with camera
- ESP8266
- toy car
- what's more, a power bank



![car](E:\study\else\RC-Raspberry-Pi-Car-with-Gesture-Control\pic\car.jpg)

> ESP8266 is at the bottom of the plastic plank.
>
> It's codes is written by Ardunio. ([code here](./Udp/Udp.ino))

## Software

Language: Python 3.6

Used packages:

- Pytorch
- OpenCV
- PyQt

Full environment setting:[here](./realtime_gesture_recog/environment.yaml)



In Raspberry site, we used this streamer: https://github.com/Five-great/mjpg-streamer, THX!

## Demo

> Poor network would limit project's performance.

[demo here]()

For friends who can't visit *Youtube*, see demo [here]()