import numpy as np
import pandas as pd
import os
import re
import argparse
import json
#keep only the no vide array in json

#usage
#python lcr_skeleton_json_to_npz.py --input C:\\Users\\dairui\\Desktop\\3d_skeleton\\Getup_p04_r01_v13_c01.json --output C:\\Users\\dairui\\Desktop\\3d_skeleton\\Getup_p04_r01_v13_c01.npz
#python lcr_skeleton_json_to_npz.py --input C:\\Users\\dairui\\Desktop\\3d_skeleton\\Drink.Fromcup_p04_r05_v17_c06.json --output C:\\Users\\dairui\\Desktop\\3d_skeleton\\Drink.Fromcup_p04_r05_v17_c06.npz
def main():
	parser = argparse.ArgumentParser(description='change .json format to .npz')
	parser.add_argument('--input', help='input .json file, as C:\\Users\\dairui\\Desktop\\3d_skeleton\\Getup_p04_r01_v13_c01.json', default=None)
	parser.add_argument('--output', help='output .npz file, as C:\\Users\\dairui\\Desktop\\3d_skeleton\\Getup_p04_r01_v13_c01.npz', default=None)
	args = parser.parse_args()
	path_input=args.input
	path_output=args.output
	#path_input="C:\\Users\\dairui\\Desktop\\3d_skeleton\\Getup_p04_r01_v13_c01.json"
	with open(path_input, 'r') as f:
		data= json.load(f)
		print(len(data["frames"]))
		#print(len(data["frames"][0][0]['pose3d']))
		#print (data["frames"][59][0]['pose3d'][1])
		#import numpy as np 
		x=[] 
		#y= np.zeros([1,39], dtype = float) 
		#print (x)
		#print (x.shape)
		#x[0][1]=123444
		#print(x)
		c=0
		for i in range(len(data["frames"])):
			temp_arr=[]
			if len(data["frames"][i])==0:
				temp_arr = [{"cumscore": 0.0,
							 "pose2d": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
										0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
							 "pose3d": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
										0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
										0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}]
				c=c+1
				y= np.zeros([1,39], dtype = float)
				#print(y)
				#print(y[0])
				for j in range(len(temp_arr[0]['pose3d'])):
					#print("1")
					y[0][j]=temp_arr[0]['pose3d'][j]
					#print(data["frames"][i][0]['pose3d'][j])
					#print(x[i][j])
					#print(x)
					#print (i,len(data["frames"][i][0]['pose3d']))
					#print(j)
				x.append(y[0])

			elif len(data["frames"][i])>0:
				c=c+1
				y= np.zeros([1,39], dtype = float)
				#print(y)
				#print(y[0])
				for j in range(len(data["frames"][i][0]['pose3d'])):
					#print("1")
					y[0][j]=data["frames"][i][0]['pose3d'][j]
					#print(data["frames"][i][0]['pose3d'][j])
					#print(x[i][j])
					#print(x)
					#print (i,len(data["frames"][i][0]['pose3d']))
					#print(j)
				x.append(y[0])
				#print(x)
			#if len(data["frames"][i])==0:
			#	x[i]=[]
		#np.savez('Getup_p04_r01_v13_c01.npz',x)
		np.savez(path_output,x)
		print ("success")
		print(c)
		#data["frames"][0][i]['pose3d'][j]
	#print (data["frames"][20][0]['pose3d'][1])
	#
	r = np.load(path_output)
	#print ("12234")
	#print(r['arr_0'].shape)
	print(r['arr_0'])
	print (len(r['arr_0']))

if __name__ == '__main__':
    main()