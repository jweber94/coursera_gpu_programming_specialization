from os.path import exists
import sys
import random
import csv
import time

def main(input_id, num_elements):
  input_lock_file_name = "input_" + input_id + ".lock"
  input_file_name = "input_" + input_id + ".csv"
  output_lock_file_name = "output_" + input_id + ".lock"
  output_file_name = "output_" + input_id + ".csv"
  # Update this and possibly the CUDA code to take as an argument for how many runs will be executed
  while True:
    ## create the input for the other program
    if not exists(input_lock_file_name):
      output_csv_line(input_file_name=input_file_name, num_elements=num_elements) # create csv
      input_lock_file = open(input_lock_file_name, 'w') # create lock file - the other cuda application can access the input data only after the lock file exists, so it is created AFTER the csv file was successfully created
      input_lock_file.close()
      # wait for the output of the other program
      while not exists(output_lock_file_name): # wait for creating the output file by the cuda application by waiting for a lock file to exist
        time.sleep(5.0)
      read_output_csv_file(output_file_name=output_file_name, num_elements=num_elements) # read out the csv file
    time.sleep(5.0) # wait before the next iteration starts - CAUTION: The data readout and print will happen as long as the cuda application does not delete the lock file (which it only does if it updates the data) - if the cuda application is not fast enough, the same data will be shown multiple times in a 5 seconds cycle 


def output_csv_line(input_file_name, num_elements):
  f = open(input_file_name,'w') # operate on the .csv file
  input_data = []
  for _ in range(num_elements):
    input_data.append(random.uniform(0, num_elements))
  # Below is based on https://www.geeksforgeeks.org/python-list-of-float-to-string-conversion/
  f.write(','.join([str(i) for i in input_data]))
  f.close()


def read_output_csv_file(output_file_name, num_elements):
  with open(output_file_name, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    output_data = next(reader)
    print(output_data)


if __name__ == "__main__":
  argc = len(sys.argv)
  if argc > 2:
    input_id = sys.argv[1]
    num_elements = int(sys.argv[2])
    main(input_id=input_id, num_elements=num_elements)
