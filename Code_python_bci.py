import numpy as np
#import matplotlib.pyplot as plt
                                        
            
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

from mne import Epochs, pick_types, events_from_annotations
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
#from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch
#from mne.datasets import eegbci
from mne.decoding import CSP 
# Importing different methods to estimate psds
# from mne.time_frequency import tfr_morlet,psd_multitaper,psd_welch
import mne
import os

# Define the directory where your .gdf files are located
data_dir = "C:/Users/laksh/Downloads/S01_MI"
# Get a list of .gdf files in the directory
gdf_files = [f for f in os.listdir(data_dir) if f.endswith('.gdf')]
# Create an empty list to store raw objects
raw_list = []
# Loop through the .gdf files and append their data to the list
for gdf_file in gdf_files:
    gdf_path = os.path.join(data_dir, gdf_file)
    raw = mne.io.read_raw_gdf(gdf_path)
    raw_list.append(raw)
# Concatenate raw objects
raw_concatenated = mne.io.concatenate_raws(raw_list)
# Extract events from annotations
raw.load_data()
raw.pick_channels(['Cz', 'C1','C2','C3','C4','C5'])
raw.crop(tmax=1)
# from sklearn.preprocessing import StandardScaler
# # Assuming raw_data is your raw EEG data
# scaler = StandardScaler()
# scaler1=StandardScaler()
# normalized_data = scaler.fit_transform(raw.get_data())
# plt.plot(normalized_data.T)  # Transpose to plot each channel separately
# plt.show()
# raw.plot_psd(fmin=1,fmax=30)
# raw.plot_psd(fmin=1,fmax=44)
# raw.plot_psd(fmin=14,fmax=24)
# raw.plot_psd(fmin=8,fmax=30)
#raw_concatenated.filter(1., 30., fir_design='firwin', skip_by_annotation='edge')
#raw_concatenated.plot_psd(fmin=1,fmax=44)
raw_concatenated.pick_channels(['Cz', 'C1','C2','C3','C4','C5'])
print(raw_concatenated.annotations)
event_id = {'1536': 1,'1537': 2,'1538': 3,'1539': 4,'1540': 5,'1541': 6}
events, _ = events_from_annotations(raw_concatenated, event_id=event_id)
                                    #dict(T1=1, T2=2))
# Define event IDs
# Create epochs
tmin, tmax = 1, 5
epochs = mne.Epochs(raw_concatenated, events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=None)

# Define picks (electrodes)
picks = pick_types(raw_concatenated.info, meg=False, eeg=True, stim=False, eog=False,
                   exclude='bads')

# Define the CSP parameters
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

# Define the LDA classifier
lda = LinearDiscriminantAnalysis()

# Create a pipeline with CSP and LDA
clf = Pipeline([('CSP', csp), ('LDA', lda)])

# Define cross-validation parameters
cv = ShuffleSplit(10, test_size=0.2, random_state=42)
# Perform cross-validation
scores = cross_val_score(clf, epochs.get_data(), epochs.events[:, -1], cv=cv, n_jobs=1)

# Print the cross-validation scores
print("Cross-Validation Scores:", scores)
print("Mean Accuracy:", np.mean(scores))
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    epochs.get_data(), epochs.events[:, -1], test_size=0.2, random_state=42)
# Fit the classifier on the entire training set
clf.fit(X_train, y_train)
# Predict labels for the test set
y_pred = clf.predict(X_test)
# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
def confusion_matrix_percent(conf_matrix):
    total_instances = np.sum(conf_matrix)
    percent_matrix = (conf_matrix / total_instances) * 100
    return percent_matrix
percent_matrix = confusion_matrix_percent(conf_matrix)
print("Confusion Matrix (Percentage Form):")
print(percent_matrix)
# Calculate overall accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Overall Accuracy:", accuracy)
# Print classification report for detailed accuracy per class
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)
#%%
raw_new_data = mne.io.read_raw_gdf("C:\\Users\\laksh\\Downloads\\S01_MI\\motorimagination_subject1_run2.gdf", preload=True)

if np.any(np.isinf(raw_new_data._data)) or np.any(np.isnan(raw_new_data._data)):
    # Handle infs and NaNs (replace them with appropriate values)
    raw_new_data._data[np.isinf(raw_new_data._data)] = np.finfo(np.float64).max
    raw_new_data._data[np.isnan(raw_new_data._data)] = 0
# Extract events from annotations in the new data

raw_new_data.load_data()
raw_new_data.pick_channels(['Cz', 'C1','C2','C3','C4','C5'])
raw_new_data.events, _ = mne.events_from_annotations(raw_new_data)

events_new_data, _ = mne.events_from_annotations(raw_new_data, event_id=event_id)

# Extract epochs around events in the new data
epochs_new_data = mne.Epochs(raw_new_data, events_new_data, event_id=event_id, tmin=tmin, tmax=tmax, baseline=None, preload=True
                        )

# Select EEG channels
eeg_epochs_new_data = epochs_new_data.pick_types(eeg=True)
MI_sigs = eeg_epochs_new_data.get_data()
# Get data from the new dataset
X_new_data = eeg_epochs_new_data.get_data()
print('trial 2')
#input_signals={1:X_new_data[3:4,:,:],2:X_new_data[6:7,:,:],3:X_new_data[4:5,:,:],4:X_new_data[12:13,:,:],5:X_new_data[:1,:,:],6:X_new_data[9:10,:,:]}
input_signals={1:MI_sigs[11:12,:,:],2:MI_sigs[:1,:,:],3:MI_sigs[8:9,:,:],4:MI_sigs[7:8,:,:],5:MI_sigs[15:16,:,:],6:MI_sigs[16:17,:,:]}
#%%
import serial
import time
#import random
# import serial.tools.list_ports
# ports = serial.tools.list_ports.comports()

# for port, desc, hwid in sorted(ports):
#         print("{}: {} [{}]".format(port, desc, hwid))

#ser_in = serial.Serial(port="COM8",baudrate = 9600,timeout = .05)
#ser_in.open()
ser_out = serial.Serial(port="COM5",baudrate = 9600,timeout = .1)
#port = "COM7"
#Port_#0002.Hub_#0001
while True:
    # Make predictions on the new data
    #value = ser_in.readline()
    #value = random.randint(1,4)
    value = int(input("Enter code: "))
    inp_num = value
    print(inp_num)
    # if value != b'':
    #     inp_num = str(value.decode())
    #     print(inp_num,type(inp_num))
    #     inp_num = int(inp_num)
    #     print(value)
    #if inp_num == 5 or inp_num == 6 or inp_num == 7 or inp_num == 8:
    #   ser_out.write(str(inp_num).encode())
        
    predictions_new_data = clf.predict(input_signals[inp_num])
   
    
    #print('trial 3')
    # Map event codes to event names
    event_names = {1: 'elbow_flexion', 2: 'elbow_extension', 3: 'supination', 4: 'pronation', 5: 'hand_close', 6: 'hand_open', 7: 'rest'}
    
    # Check if specific events exist in the predictions and send corresponding signals through serial port
    
    # Configure the serial port
    #ser=serial.Serial(port='COM5', baudrate=115200,timeout=0.1)
    #ser = serial.Serial(port='COM10', baudrate=115200, timeout=0.1)
    #print('taylor')
    for event_code in event_id.values():
        if event_code in predictions_new_data:
            event_name = event_names[event_code]
            signal = event_code  # You can change this to whatever signal you want to send
            ser_out.write(str(signal).encode())
            #ser_out.write(bytes(signal,'utf-8'))
            
            print(f"The event (name): {event_name} (code: {event_code}) exists in the predictions and signal {signal} has been sent.")
        else:
            print(f"The event (code: {event_code}) does not exist in the predictions.")
    time.sleep(1)
# Close the serial port
#%%
#ser_in.close()
ser_out.close()
#%%
# plt.plot(input_data[1][0][0])
# plt.plot(input_data[2][0][0])
# plt.plot(input_data[3][0][0])
# plt.plot(input_data[4][0][0])
# plt.plot(input_data[5][0][0])
# plt.plot(input_data[6][0][0])

#%%%
# >>> with serial.Serial('/dev/ttyS1', 19200, timeout=1) as ser:
# ...     x = ser.read()          # read one byte
# ...     s = ser.read(10)        # read up to ten bytes (timeout)
# ...     line = ser.readline()   # read a '\n' terminated line
