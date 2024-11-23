import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import csv

mpl.use('tkagg')

fig, ax = plt.subplots()

su_x = []
su_y1 = []
su_y2 = []
su_y3 = []

with open('/home/amir/repos/FGPRS/results/speedup/convolution.csv', 'r') as file:
	dummy = csv.reader(file)

	for row in dummy:
		su_x.append(row[0])
		su_y1.append(row[1])
		su_y2.append(row[2])
		su_y3.append(row[3])
	
plt.bar(su_x, su_y1)
plt.show()
plt.show()
# results/speedup/convolution.csv
# /home/amir/repos/FGPRS/results/speedup/convolution.csv