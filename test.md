# MAE345/543: Final Project

For the final project in this class, you will write a program that guides the
Crazyflie from one end of the netted area in G105 to the other. To navigate, each Crazyflie is outfitted with a PoV camera capable of live streaming a view from the Crazyflie to your computer. This project is _open ended_, and you are free to use any approach to complete the task at hand.

Unfortunately, due to compatibility issues between the video encoder and VirtualBox, we will not be able to use the virtual Ubuntu install that we have worked with thus far. One member of your group will have to install the software necessary to interface with the Crazyflie on a computer with at least two USB ports. While the setup described below should work for any operating system, the instructors have found that the video encoder produces streams with a lower frame rate on MacOS. It is also slightly more complicated to install `cfclient` on MacOS. Therefore, we suggest using a computer with either Windows or Linux (we tested the software on Ubuntu 18.04).

## Setting Up The Software

If at least one person in the group does not have Python installed on their laptop, you will need to do so. We recommend using [Anaconda](https://www.anaconda.com/) to install Python. Anaconda is a suite of software used to manage different versions of Python and various libraries you might use. Please make sure to install the Python 3.7 version of Anaconda linked [here](https://www.anaconda.com/distribution/). Once installed complete the following steps:

1. Depending on your operating system, do either:

  a. **(MacOS / Linux)** Open your terminal. If you installed Anaconda using a terminal, you will have to close and reopen your terminal for its installation to take effect. You should see `(base)` written next to your command prompt.

  b. **(Windows)** Open the application named "Anaconda Prompt". You should see `(base)` written next to the command prompt.

2. To install a package, use the command `conda install <package name>`. You will need the following packages: `numpy`, `opencv`, `jupyter`.

3. You will also need to install `cflib`. To do so, download the source code from [GitHub](https://github.com/bitcraze/crazyflie-lib-python). In the Anaconda prompt, navigate to the directory where you have downloaded this source code (using `cd`) and run the command `pip install -e .`

  3.b **(Linux)** If you are installing cflib on linux, you need to update the UDEV rules to recognize the Crazyflie Radio. There are step by step instructions at the bottom of the GitHub page. After completing them, you may need to reboot your computer.

4. Install `cfclient`. To do so, download the source code from [this](https://github.com/bitcraze/crazyflie-clients-python) GitHub repository. Instructions for installing `cfclient` are presented in the GitHub repository, and you may disregard the instructions for windows about creating an installer.

Once you have installed `cfclient`, you can run it by entering the command `cfclient` into your shell. You will have to configure your drone in the same way as Lab 1 again as they were mixed up while the cameras were added and yours may now have a different group number.

## Setting Up The Hardware

In addition to the Crazyflie and newly attached camera, the instructors will be providing you with three more hardware components:

- A power supply (wall wart).

- An analog video receiver.

- An analog-to-digital video encoder.

Please set the power supply to 12V by adjusting the orange selector with a flat head screwdriver or similar implement and attach the larger of the yellow tips. This device will be used to power the analog receiver.

Next, you need to set the camera and receiver to operate on the same channel. Channels are identified by a letter and number, e.g. A3 or B1. To change the receiver's channel, simply hit the the CH button. To change the camera's channel, press the button on the camera quickly to change the number or hold it down to change the letter. On the camera, the blue LED indicates the letter and the red LED indicates the number. Putting these in the left most position indicates channel A1. Finally, plug the receiver into your computer.
If you want to test this setup without using Python, you can try to record a video using VLC (Windows / Linux) or QuickTime (MacOS). There is an example Python script discussed in the next section that can also be used to test your hardware setup.


## Test Scripts

Along with these instructions, the instructors have provided two example Python scripts. If you are new to Python, these scripts are more like conventional programs you may have written in the past than the Jupyter notebooks we used throughout the semester in that all the code in them is executed at once. Both can be run by entering `python <scriptname>` in your terminal.

The script `test_video.py` uses OpenCV to open a video device on your computer and plays frames from it in a computer. Depending on your computer, the video encoder may be video device 0 or 1 (your webcam might be assigned to the other number). By default, the script reads frames from video device 0, so you may need to change it to 1 manually.

The script `test_cf.py` is a minimal implementation of the full control loop of the Crazyflie. It does the following:

1. Connects to the Crazyflie.

2. Reads frames from the video encoder for five seconds. This removes any black frames that may be received while the radio searches for the camera's frequency.

3. Ascends from the ground.

4. Reads a frame and processes with OpenCV. The processing is broken into the following steps:

  a. Converts the frame from BGR (OpenCV's default) to HSV, which is a more convenient color space for filtering by color.

  b. Applies a mask that creates an image where a pixel is white if the corresponding pixel in the frame has an HSV value in a specified range and black otherwise. The color range was tuned by the instructors to match the red of the obstacles, but different lighting conditions and cameras may require further tuning.

  c. Applies OpenCV's contour detection algorithm to the masked image. This finds the edges between regions of color in an image.

  d. Checks the area (in pixels squared) of the largest contour. If it's greater than a threshold, the Crazyflie is instructed to move right and start Step 4 over. Otherwise, we move on to step 5.

5. The Crazyflie is instructed to land.

Thus, this script causes the drone to move right until it no longer sees a red object of significant size. To run this script, you will need to change the URI in the file to match your drone / group number.

## Lab Setup

The lab setup is very similar to the previous lab. There are a number of PVC pipe obstacles now painted red in G105. You are free to move them around for testing purposes but do not remove them from G105. In addition, the battery life is significantly shorter for the modified Crazyflies. This is due to the fact that the camera and added circuitboard are close to the Crazyflie's maximum load and that the camera is using the Crazyflie's battery to transmit video. To help manage this, the instructors will be placing additional batteries and a battery charger in G105.

## Demo Day and Grading

## Some Suggested Approaches

One option is to construct an estimate of obstacle locations in order to build a map, and then use a sampling based planner to navigate toward the goal region. You will need to periodically recompute your motion plan as you gather more information about obstacle locations.

Another option is to do what is known as reactive planning. In reactive planning, you compute a control action to take based on the current context of the system. Notably, reactive planners often do not need a complete map to function. While there are many ways you could implement a reactive planner, optical flow is particularly useful tool for this approach. A simple planner might steer the drone toward regions of it's visual field where time to collision is high.

## Advice From The Instructors

- Don't reinvent the wheel. Unlike previous assignments, you are not restricted in the libraries and techniques you may use to approach this challenge. We especially recommend that you use OpenCV to simplify things as much as possible. For example, if you want to use optical flow to compute the time to collision of your drone with an obstacle, OpenCV has a very good implementation of the optical flow algorithm. Similarly, the script `test_cf.py` makes good use of OpenCV's contour detection algorithm. You are welcome to look up documentation / information on the use of OpenCV, Numpy, etc.

- Start simple. Begin with just moving around one obstacle. Once that is working reliably, add a second and a third.

- Start early. As you may have learned during the first two hardware labs, getting robots to work in the real world is tricky, and this assignment will take time. The instructors will also not be able to provide very much help if you start the night before the deadline.
