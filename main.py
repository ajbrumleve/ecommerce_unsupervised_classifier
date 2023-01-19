# This is a sample Python script.
import pickle
import logging
import numpy as np

import create_models


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def add_session(cust_id, category_list: list):
    cust_array = cust_arrays[cust_id]
    arr = np.append(cust_array, np.zeros(int((len(cust_arrays[list(cust_arrays.keys())[0]]) - 1) / 2)))
    for cat in category_list:
        cat_id = cat_map[cat]
        adj_cat_id = len(cust_arrays[list(cust_arrays.keys())[0]]) + cat_id
        arr[adj_cat_id] += 1
    prediction = int(models.svm_model.predict(arr.reshape(1, -1))[0])
    prob = models.svm_model.predict_proba(arr.reshape(1, -1))[0][prediction]
    print(f"The customer class of this customer is {prediction}. This is {prob * 100}% likely.")
    return models.svm_model.predict(arr.reshape(1, -1))[0]


if __name__ == '__main__':
    response = False
    while not response:
        train_bool = input("Do you want to train the model? Y/N ")
        if train_bool == "Y":
            models = create_models.train_model()
            response = True
        elif train_bool == "N":
            try:
                with open('models/model_info.pkl', "rb") as f:
                    models = pickle.load(f)
                    f.close()
                response = True
            except FileNotFoundError:
                logging.error("There is no pretrained model. Please run again and build new model.")

        else:
            print("Please enter either Y or N")
    km_model = models.km_model
    svm_model = models.svm_model
    cust_arrays = models.cust_arrays
    cat_map = models.cat_map
    customer_id = input('What is the customer id?')
    cat_list = []
    while True:
        cat_entry = input('What is the category to add? If no more events, enter "Finished"')
        if cat_entry == "Finished":
            break
        else:
            cat_list.append(cat_entry)
    print(add_session(customer_id, cat_list))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
