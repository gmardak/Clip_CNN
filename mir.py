import requests, time
import RPi.GPIO as GPIO

time.sleep(120)
#print('start')
output_signal = 12
input_signal = 16
GPIO.setmode(GPIO.BCM)
GPIO.setup(output_signal, GPIO.OUT)
GPIO.setup(input_signal, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

ip = '10.0.0.185'
host = 'http://' + ip + '/api/v2.0.0/'

token = 'Basic YXJkYWtrOjdiZmMwNzk1ZDdkZmQ1NGMxNzdlNTJlZjAxZWIwZjk2YWU1MDJkNGE1ZGVhMGQ1OGNlZDA5OGYyMDJhYjgzNjE='
headers = {
    'accept': 'application/json',
    'Authorization': token,
    'Content-Type': 'application/json',
    'Accept-Language': 'en_US'
}

state = 1

def set_low():
    GPIO.output(output_signal, GPIO.LOW)
    
def set_high():
    GPIO.output(output_signal, GPIO.HIGH)
    
def get_register_value():
    response = requests.get(host + 'registers/1', headers=headers)
    if response.status_code == 200:
        #print(response)
        return response.json()['value']
    else:
        return False

def set_register_value():
    data = {
        'value': 6
        }
    response = requests.put(host + 'registers/1', headers=headers, json=data)
    #print(response)
    if response.status_code == 200:
        return True
    else:
        return False
  
while True:

    if state == 1:
        reg_value = get_register_value()
        if reg_value == 5:
            #print('REG Value == 5 entered if')
            set_high()
            time.sleep(1)
            set_low()
            state = 2
        #print('state = 1 and reg_value: ' + str(reg_value))
            
    elif state == 2:
        reg_value = get_register_value()
        #print('state = 2 and reg_value: ' + str(reg_value))
        if GPIO.input(input_signal) == GPIO.HIGH:
            status = set_register_value()
            #print('entered if and GPIO High and status: ' + str(status))
            if status:
                state = 1
                #print('entered another if')
                continue
        elif reg_value != 5:
            #print('entered elif and reg_value: ' + str(reg_value))
            state = 1
            
    time.sleep(0.5)