import csv
import random
import numpy as np
from feast import *

def load_data():
	with open('all_combined.csv', 'rb') as f:
		data = []
		reader = csv.reader(f)
		for i,row in enumerate(reader):
			if i == 0:
				row = row[4:]
			if i > 0:
				row = row[4:]
				data.append([float(cell) for cell in row])
	return data

def alter_data():
	with open('all_combined.csv', 'rb') as f:
		with open('all_combined2.csv','wrb') as o:
			writer = csv.writer(o)
			reader = csv.reader(f)
			data = []
			for i,row in enumerate(reader):
				row = row[4:]
				if i == 0:
					row.append("label")
				else:
					row.append("positive" if random.random() > 0.5 else "negative")
				#if i > 100:
				#	break
				print row
				data.append(row)

			#print data
			#print len(data)
			writer.writerows(data)



def main():
	data = alter_data()
	#raw_data = load_data()
	
	#data = np.array(raw_data)
	#print (data[0][0] + 1)
	#labels = np.array([float(random.randint(0,2)) for x in range(len(data))])
	
	#cife_rankings = CIFE(data, labels, len(data[0]))
	

main()