import requests, json, time
print('start')

ip = 'mir.com'
# ip = '10.0.0.139'
host = 'http://' + ip + '/api/v2.0.0/'

token = 'Basic YXJkYWtrOjdiZmMwNzk1ZDdkZmQ1NGMxNzdlNTJlZjAxZWIwZjk2YWU1MDJkNGE1ZGVhMGQ1OGNlZDA5OGYyMDJhYjgzNjE='
headers = {
    'Authorization': token,
    'Content-Type': 'application/json'
}
get_registers = requests.get(host + 'registers', headers=headers)
missions = json.loads(get_registers.text)
for register in get_registers:
    print(register)

# get_missions = requests.get(host + 'missions', headers=headers)
# missions = json.loads(get_missions.text)
# mission_name = 'API_test'
# for mission in missions:
#         if mission['name'] == mission_name:
#             mission_id = mission['guid']

# print(mission_id)

# mission_api_message = {
#   "mission_id": mission_id,
#   "priority": 0,
# }

# post_mission = requests.post(host + 'mission_queue', headers=headers, data=json.dumps(mission_api_message))

# if post_mission.status_code == 201:

#     print("Mission added to queue successfully.")
#     state_id = {'state_id' : 3}
#     start_mission = requests.put(host + 'status', headers = headers, json = state_id)
#     # print('start_mission status: ' + start_mission.status_code)
#     # get_mission_actions = requests.get(host + 'missions/'+mission_id+'/actions', headers=headers)
#     # actions = json.loads(get_mission_actions.text)
#     # action_type = 'set_plc_register'
#     # for action in actions:
#     #     if action['action_type'] == action_type:
#     #         for id in action['parameters']:
                
#     #             if id['id'] == 'value':
#     #                 print(id)
#     #                 # print(id['guid'])
#     #                 plc_set = {'value' : 1}
#     #                 # requests.put(host + 'missions/' + mission_id + '/actions/' + id['guid'], headers = headers, json = plc_set)
#     #                 time.sleep(30)
#     #                 plc_register_id = '1'
#     #                 requests.put(host + '/registers/' + plc_register_id, headers = headers, json = plc_set)
#     #                 print(id)
#     #                 break
# else:
#     print(f"Failed to add mission to queue. Error: {post_mission.text}")


