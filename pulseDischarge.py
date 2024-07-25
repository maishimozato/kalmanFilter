#script for pulse discharge test with PEL-3031E through USB remote control
import pyvisa
import time
import csv

rm = pyvisa.ResourceManager('@py')
print(rm.list_resources)