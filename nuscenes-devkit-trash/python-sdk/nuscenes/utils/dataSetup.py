import json






files = ['train', 'val', 'test']

def data_collector(file):


    # List to store name tokens
    train = []

    # Open the JSON file
    with open(f'/cvrr/bevfusion_bjork/tools/json_map/{file}.json', 'r') as json_file:
        data = json.load(json_file)

    # Search for name tokens in the JSON data
    for item in data:
        if 'name' in item:
            train.append(item['name'])
            

    # Print the list of name tokens
    print("\n List of name tokens in the train file:", train)

    print(len(train))

    return train


    # # List to store name tokens
    # train = []
    # val = []
    # test = []

    # for file in files:
    #     # Open the JSON file
    #     with open(f'/cvrr/bevfusion_bjork/tools/json_map/{file}.json', 'r') as json_file:
    #         data = json.load(json_file)

    #     # Search for name tokens in the JSON data
    #     for item in data:
    #         if 'name' in item:
    #             if file == 'train':
    #                 train.append(item['name'])
    #             elif file == 'val':
    #                 val.append(item['name'])
    #             else:
    #                 test.append(item['name'])

    # # Print the list of name tokens
    # print("\n List of name tokens in the train file:", train)
    # print("\n List of name tokens in the val file:", val)
    # print("\n List of name tokens in the test file:", test)
    # print(len(train))
    # print(len(val))
    # print(len(test))