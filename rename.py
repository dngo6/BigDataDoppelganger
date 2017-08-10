import os

count = 0
for file_name in os.listdir("2_All"):
	file_name = "2_All/" + file_name
	new_name = "2_All/" + str(count) + ".jpg"
	os.rename(file_name, new_name)
	count = count + 1

