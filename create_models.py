import logging
import numpy as np
import sys
import timeit
import pickle
import warnings

from datetime import datetime
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC


class Models:
    def __init__(self):
        self.km_model = KMeans()
        self.svm_model = SVC()
        self.cat_map = {}
        self.cust_arrays = {}


def train_model():
    logger = logging.getLogger("test")
    logging.basicConfig(level=logging.DEBUG)
    log_info = logging.FileHandler('test-log.log')
    log_info.setLevel(logging.INFO)
    logging.getLogger('').addHandler(log_info)

    num_lines = 5000000
    knee = None
    target_i = 30

    def get_cat_num(x):
        return cat_map[x]

    t_file = timeit.default_timer()
    logging.info(datetime.now().strftime('%H:%M:%S.%f') + " - " + "Reading through data first time")
    cat_map = {}
    with open('data/2019-Oct.csv', 'r') as f:
        next(f)
        line_num = 0
        for line in f:
            if line_num == num_lines:
                break
            split_line = line.split(",")
            cat = split_line[4]
            del split_line
            if cat == "":
                continue
            if cat not in cat_map.keys() and len(cat_map) == 0:
                cat_map[cat] = 0
            elif cat not in cat_map.keys():
                cat_map[cat] = len(cat_map)
            line_num += 1
    logging.info(datetime.now().strftime(
        '%H:%M:%S.%f') + " - " + f"First cycle through data in {timeit.default_timer() - t_file} seconds")
    del f, line
    num_cats = len(cat_map)

    t_section = timeit.default_timer()
    logging.info(datetime.now().strftime('%H:%M:%S.%f') + " - " + "Reading through data second time")
    cust_arrays = {}
    session_arrays = {}
    with open('data/2019-Oct.csv', 'r') as f:
        next(f)
        line_num = 0
        for line in f:
            if line_num == num_lines:
                break
            split_line = line.split(",")
            event, cat, user, session = (split_line[1], split_line[4], split_line[7], split_line[8])
            del split_line
            if cat == "":
                continue
            # if session not in session_arrays.keys():
            #     first_event[(user, cat)] = 0

            cust_arrays[user] = cust_arrays.get(user, np.zeros(num_cats * 2 + 1))
            session_arrays[(user, session)] = session_arrays.get((user, session),
                                                                 [cust_arrays[user], np.zeros(num_cats)])
            arr = cust_arrays[user]
            arr2 = session_arrays[(user, session)]
            if event == "view":
                arr[get_cat_num(cat)] += 1
                arr2[1][get_cat_num(cat)] += 1
            if event == "purchase" or event == "cart":
                arr[get_cat_num(cat) + num_cats] += 1
                arr2[1][get_cat_num(cat)] += 1
            line_num += 1
            del arr
    del cat, event, f, line, line_num, num_lines, user
    logging.info(datetime.now().strftime(
        '%H:%M:%S.%f') + " - " + f"Second run through data in {timeit.default_timer() - t_section} seconds")

    logging.info(datetime.now().strftime('%H:%M:%S.%f') + " - " + "Creating customer array")
    t_section = timeit.default_timer()
    cust_array_vals = cust_arrays.values()
    cust_array_vals_lst = list(cust_array_vals)
    del cust_array_vals
    cust_array = np.array(cust_array_vals_lst)
    del cust_array_vals_lst
    logging.info(datetime.now().strftime(
        '%H:%M:%S.%f') + " - " + f"Customer array created in {timeit.default_timer() - t_section} seconds")

    logging.info(datetime.now().strftime('%H:%M:%S.%f') + " - " + "Creating customer id array")
    t_section = timeit.default_timer()
    cust_array_keys = cust_arrays.keys()

    cust_array_keys_lst = list(cust_array_keys)
    del cust_array_keys
    cust_ids = np.array(cust_array_keys_lst)
    del cust_array_keys_lst
    logging.info(datetime.now().strftime(
        '%H:%M:%S.%f') + " - " + f"Customer id array created in {timeit.default_timer() - t_section} seconds")

    logging.info(datetime.now().strftime('%H:%M:%S.%f') + " - " + "Beginning normalization of customer array")
    t_section = timeit.default_timer()
    total_events_per_cust = cust_array.sum(axis=1)
    cust_array /= cust_array.sum(axis=1, keepdims=True)
    cust_array = np.append(cust_array, total_events_per_cust[:, np.newaxis], axis=1)
    del total_events_per_cust
    mean_num_events = np.mean(cust_array[:, -1])
    std_num_events = np.std(cust_array[:, -1])
    cust_array[:, -1] = (cust_array[:, -1] - mean_num_events) / std_num_events
    logging.info(datetime.now().strftime(
        '%H:%M:%S.%f') + " - " + f"Customer array normalized in {timeit.default_timer() - t_section} seconds")

    logging.info(datetime.now().strftime('%H:%M:%S.%f') + " - " + "Choosing number of knees with Elbow Method")
    t_section = timeit.default_timer()
    inertias = []
    cluster_ct = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                  29,
                  30]
    # for i in cluster_ct:
    #     model = KMeans(n_clusters=i,init='k-means++')
    #     model.fit(cust_array)
    #     inertias.append(model.inertia_)
    #
    # kn = KneeLocator(cluster_ct, inertias, curve='convex', direction='decreasing')
    # knee = kn.knee
    # logging.info(datetime.now().strftime('%H:%M:%S.%f')+" - "+f"Elbow method took {timeit.default_timer()-t_section}
    # seconds")
    # print(f"The elbow method gives {knee} as the ideal number of clusters")

    for i in cluster_ct:
        if knee is not None and target_i == 30:
            target_i = i + 2
            logging.info(datetime.now().strftime('%H:%M:%S.%f') + " - " + f"Will check until {target_i} clusters")
        if i < target_i:
            model = KMeans(n_clusters=i, init='k-means++')
            model.fit(cust_array)
            inertias.append(model.inertia_)
            logging.info(
                datetime.now().strftime('%H:%M:%S.%f') + " - " + f"With {i} clusters, the inertia is {model.inertia_}")
            del model
            if i >= 2:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    kn = KneeLocator(cluster_ct[:i], inertias, curve='convex', direction='decreasing')
                    knee = kn.knee
        else:
            break
    del cluster_ct, inertias, i, target_i, kn
    logging.info(
        datetime.now().strftime(
            '%H:%M:%S.%f') + " - " + f"Elbow method took {timeit.default_timer() - t_section} seconds")
    print(f"The elbow method gives {knee} as the ideal number of clusters")

    logging.info(datetime.now().strftime('%H:%M:%S.%f') + " - " + "Beginning to train model")
    t_section = timeit.default_timer()
    model = KMeans(n_clusters=knee, init='k-means++')
    model.fit(cust_array)
    labels = model.labels_
    logging.info(
        datetime.now().strftime(
            '%H:%M:%S.%f') + " - " + f"Model trained in {timeit.default_timer() - t_section} seconds")

    # logging.info(datetime.now().strftime('%H:%M:%S.%f')+" - "+f"Creating customer group dict") t_section =
    # timeit.default_timer() cust_group_dict = {} for i in range(knee): logging.info(datetime.now().strftime(
    # '%H:%M:%S.%f')+" - "+f"Working on group {i}") arr1 = cust_ids[labels == i].reshape(-1,1) cust_group_dict[i] = arr1
    # del arr1 del cust_array, cust_ids, labels, knee, i logging.info(datetime.now().strftime('%H:%M:%S.%f')+" -
    # "+f"Customer group dict created in {timeit.default_timer()-t_section} seconds")

    logging.info(datetime.now().strftime('%H:%M:%S.%f') + " - " + "Creating session array")
    t_section = timeit.default_timer()

    new_arr = np.zeros((len(session_arrays), num_cats*3+2))
    session_arrays_keys_lst = list(session_arrays.keys())
    session_arrays_vals_lst = list(session_arrays.values())
    session_arrays_keys_arr = np.array(session_arrays_keys_lst)
    session_arrays_vals_arr = np.array(session_arrays_vals_lst)
    session_cust_arr = session_arrays_vals_arr[:,0]
    session_ses_arr = session_arrays_vals_arr[:,1]
    user_arr = session_arrays_keys_arr[:,0]
    user_label_dict = dict(zip(cust_ids, labels))
    user_arr = np.vectorize(user_label_dict.get)(user_arr)
    for i in range(len(session_arrays)):
        new_arr[i] = np.concatenate((session_cust_arr[i].reshape(1,-1),session_ses_arr[i].reshape(1,-1),user_arr[i].reshape(1,-1)),axis=1)
    logging.info(datetime.now().strftime(
        '%H:%M:%S.%f') + " - " + f"Session array created in {timeit.default_timer() - t_section} seconds")

    logging.info(datetime.now().strftime('%H:%M:%S.%f') + " - " + "Testing support vector machine model")
    t_section = timeit.default_timer()
    xTrain, xTest, yTrain, yTest = train_test_split(new_arr[:, :-1], new_arr[:, -1])
    # svm_model_linear = SVC(kernel='linear', C=1, probability=True).fit(xTrain, yTrain)
    svm_model_linear = LinearSVC(dual=False, class_weight='balanced', verbose=2).fit(xTrain, yTrain)
    svm_predictions = svm_model_linear.predict(xTest)
    # model accuracy for xTest
    accuracy_svm = svm_model_linear.score(xTest, yTest)
    print("svm accuracy - ", accuracy_svm)
    # creating a confusion matrix
    cm_svm = confusion_matrix(yTest, svm_predictions)
    print("Confusion matrix - ", cm_svm)

    logging.info(
        datetime.now().strftime(
            '%H:%M:%S.%f') + " - " + f"SVM model tested in {timeit.default_timer() - t_section} seconds")

    del t_section

    # for key in cust_group_dict.keys():
    #     key_size = sys.getsizeof(cust_group_dict[key])
    #     np.save(f"files/group_{key}",cust_group_dict[key])
    #     print(f"The size of array{key} in dictionary is {sys.getsizeof(cust_group_dict[key])} bytes")
    #
    logging.info("The size of the Kmeans model is {} bytes".format(sys.getsizeof(pickle.dumps(model))))
    logging.info("The size of the svm model is {} bytes".format(sys.getsizeof(pickle.dumps(svm_model_linear))))

    # save the model to disk
    model_filename = 'models/kmeans.mdl'
    session_model_filename = 'models/session_svm.mdl'
    try:
        pickle.dump(model, open(model_filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    except FileNotFoundError:
        logging.error("No models directory")
        return
    try:
        pickle.dump(svm_model_linear, open(session_model_filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    except FileNotFoundError:
        logging.error("No models directory")
        return
    logging.info(datetime.now().strftime('%H:%M:%S.%f') + " - " + f"Models saved")

    logging.info(
        datetime.now().strftime('%H:%M:%S.%f') + " - " + f"Total time - {timeit.default_timer() - t_file} seconds")
    models = Models()
    models.km_model = model
    models.svm_model = svm_model_linear
    models.cust_arrays = cust_arrays
    models.cat_map = cat_map
    try:
        pickle.dump(models, open('models/model_info.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        logging.info("The size of the models object is {} bytes".format(sys.getsizeof(pickle.dumps(models))))

    except FileNotFoundError:
        logging.error("No models directory")
        return

    return models

