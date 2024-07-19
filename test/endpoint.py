import requests
import json

# Get the RestEndpoint and PrimaryKey and attach to the scoring_uri and key
scoring_uri = 'http://d26fdb76-c5cf-475b-954a-709187f044c0.southeastasia.azurecontainer.io/score'
key = 'XeZ59SwDjK6IT2wKdsNLRZGhcS6jVMRC'

# Two sets of data to score, so we get two results back
data = {
  "data": [
        {
            "age": 70, 
            "anaemia": False, 
            "creatinine_phosphokinase": 92, 
            "diabetes": False, 
            "ejection_fraction": 60, 
            "high_blood_pressure": True, 
            "platelets": 317000, 
            "serum_creatinine": 0.8, 
            "serum_sodium": 140, 
            "sex": False, 
            "smoking": True, 
            "time": 74
        }
    ],
    "method": "predict"
}

# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())
