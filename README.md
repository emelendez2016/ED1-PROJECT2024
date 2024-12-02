# ED1-PROJECT2024 ----
This repository contains the IPYNB file for our Harmful Brain activity classification project

# To do:
Please when making changes to the ipynb also  update ReadMe file as well with the work  you just completed and adding the Commit detail and Description.

## The following implementations steps are done [At time of pulling]:

PRE PROCESSING: (Estuardo Melendez - Manny)
- [X] Downloaded Dataset 
- [X] Remove Noise - filter [Notch Filter removes 60hz interference for powerline noise] !CAN BE ADJUSTED ASK PROFESSOR!!
- [X] Artifact Removal - Applied ICA for Artifact removal (Can be tuned for diffferent target !! ask professor!! )
- [X] Signal normalization  - Since all plots vary in amplitude we used Numpy to normalize signals !! ASK  FOR OPINION FROM ALI
- [X] Time Frequency Optimization -- Frequency resolution up to 100hz, Time segments of nperseg =128 for EEG with 200Hz sampling !! ASK ALI 

Feature Extraction : (Estuardo Melendez- Manny )

- [X] Accessed pre-processed parquet files
- [X] Divided spectogram into 2 second intervals for better feature extraction 
- [X] Extracted time domain features : vairance, skewness, rms and power for each EEG Frequency bands 
- []
- []

!!! PLEASE ASK SPONSOR FOR FEEDBACK!!!!!!


Machine Learning : (Estuardo Melendez - Kevin)

- []
- []
- []
- []
- []

Deep Learning: (Michael - Intisarul)

- []
- []
- []
- []
- []

UI: (Michael - Intisarul)

- [x] included text file "ideas for prediction stats.txt" for everyone to review regarding linking module outputs to UI
- [x] included sample idea for ui format: "working UI.py"
- []
- []
- []
 
