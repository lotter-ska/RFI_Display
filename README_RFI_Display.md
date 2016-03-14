# RFI_Display
RFI monitor(Ratty) data processing

Downloads and process data from Ratty receiver in ASC on site.  Output plot to png images.

Data server address:
http://rfimonitor.kat.ac.za/rfi_data/

Folder on display server:
/home/reverb-chamber/Desktop/Ratty
(Png's are output there and required file must be in this folder)

Required files:
occupancy.npy - 48h rolling occupancy data file
Cul_db.npy - culprit database file.  Data  generated from RFI_Cul_data1.py script.
RFI_mask.npy - Self generated RFI mask DB

