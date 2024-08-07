# EEG-Based Motor Imagery Control System
This project explores the potential of EEG-based motor imagery (MI) control for individuals with paralysis. By translating brainwave patterns into control signals, this system could provide assistive technology to enhance independence and quality of life.

## Features:

* **Motor Imagery Classification:**  Uses machine learning to classify EEG signals associated with different motor imagery tasks (e.g., imagining hand movements).
* **Device Control:**  Generates control signals that can be sent to an Arduino microcontroller to control connected devices (such as a fan, light bulb, or a servo motor).

## Requirements:

* Python 3.x
* MNE-Python (https://mne.tools/stable/index.html) 
* Scikit-learn (https://scikit-learn.org/stable/)
* Serial (https://pypi.org/project/pyserial/)
* Arduino IDE

## How to Use:

1. **Install Dependencies:**  Install the required libraries 
2. **Download EEG Data:** Download the EEG dataset from [(http://bnci-horizon-2020.eu/database/data-sets)] Take Sl.no 25 and an individual file from it
4. **Run the Script:** Run the `Code_python_bci.py` script to:
    * Train a machine learning model.
    * Classify new EEG data.
    * Generate control signals to send to an Arduino.
5. **Run the Automation script on Arduino IDE **

## Contributions:

Contributions to this project are welcome! You can:
* Improve the machine learning algorithms.
* Add support for new devices or control schemes.
* Optimize the code for efficiency or performance.

## Disclaimer:

This project is for educational and experimental purposes only. It is not intended for medical use or any application that requires high accuracy or reliability. Please consult a qualified medical professional for any healthcare-related matters. 
