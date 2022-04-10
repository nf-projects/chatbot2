import json

# Json Editor to edit the intents
# not working atm

filename = "./intents.json"

def choices():
    print("")
    print("")
    print("Select JSON Edit Type:")
    print("(1) View Data")
    print("(2) Edit Data")
    print("(3) Exit")

    choice = input("Enter Choice: ")
    if choice == "1":
        print("1")
        view_data()
    elif choice == "2":
        print("2")
    elif choice == "3":
        print("3")
    else:
        print("Invalid input")  

# function to add to JSON
def write_new_json(new_data, filename='./intents.json'):
	with open(filename,'r+') as file:
		# First we load existing data into a dict.
		file_data = json.load(file)
		# Join new_data with file_data inside emp_details
		file_data["intents"].append(new_data) 
		# Sets file's current position at offset.
		file.seek(0)
		# convert back to json.
		json.dump(file_data, file, indent = 4)

def update_json(new_data, filename='./intents.json'):
	with open(filename,'w') as file:
		# First we load existing data into a dict.
		file_data = json.load(file)
		# Join new_data with file_data inside emp_details
		file_data["intents"].update(new_data) 
		# Sets file's current position at offset.
		file.seek(0)
		# convert back to json.
		json.dump(file_data, file, indent = 4)    

# python object to be appended
new = {
    "tag": "compliment",
	"patterns": ["nice", "cool"],
	"responses": ["thanks broski", "ty"]
	}

update = {
    "tag": "compliment",
	"patterns": ["nice", "epic"],
	"responses": ["thanks broski", "ty"]
	}    

#write_new_json(new)

update_json(update)