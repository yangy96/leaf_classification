from end_classifiers.parse_data import parse_data, convert_representation
from end_classifiers.model import end_classifier_model, feature_augmented_model, SVM_classifier
from end_classifiers.model import *
from split_train_test import split_data
import os
import sys
import json
from shutil import copyfile

verbosity = True

class node_classifiers :
    def __init__(self, memory_folder):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.memory_dir = memory_folder
        self.training_epochs = 40

    def train_node_classifiers(self):
        self.training_data = parse_data(self.memory_dir, weighted = False)

        for each_memory in self.training_data.keys():
            if verbosity :
                print("Currently training for memory name - ", each_memory)
                print("Current class weight - ", self.training_data[each_memory]["weight"])

            memory_dir = os.path.join(self.memory_dir, each_memory)

            current_classifier = end_classifier_model(model_dir =  memory_dir, \
            logit_mapping = self.training_data[each_memory]["logit_mapping"],\
            weights =  list(self.training_data[each_memory]["weight"].values()))

            training_accuracy = current_classifier.train_model(self.training_data[each_memory]["samples"], self.training_epochs)

            # Saving the logit to class names file

            fp = open(os.path.join(memory_dir, "logit_mapping.json"), "w")
            json.dump(self.training_data[each_memory]["logit_mapping"], fp)
            fp.close()

    def evaluate_test(self, memory_mapping_file, test_folder, class_list):
        fp = open(memory_mapping_file, "r")
        memory_distances = json.load(fp)
        fp.close()

        test_files = split_data.read_training_data(test_folder)
        print("No of test files - ", len(test_files))
        memory_wise_files = {}

        ood_files = []

        # Evaluate the memory models first, this is because otherwise the individual classifiers
        # have to be loaded on-off again, causing slowdown

        for each_item in test_files:
            image_name = each_item["image"]

            if image_name in memory_distances.keys():
                all_distances = memory_distances[image_name]["distances"]
                ground_truth_class_name = memory_distances[image_name]["orig_data"]["class_name"]
                features = memory_distances[image_name]["orig_data"]["features"]

                nearest_memory = None
                closest_distance = np.inf

                # Right now it finds the nearest memory

                for each_memory in memory_distances[image_name]["distances"].keys():
                    # memory_distance = memory_distances[image_name]["distances"][each_memory]["distance"]
                    memory_distance = memory_distances[image_name]["distances"][each_memory]
                    # threshold = memory_distances[image_name]["distances"][each_memory]["threshold"]

                    if memory_distance < closest_distance :
                        closest_distance = memory_distance
                        nearest_memory = each_memory

                if nearest_memory is not None :

                    new_info = {"image":image_name, "class_name" : ground_truth_class_name, "features" : features}

                    if nearest_memory in memory_wise_files.keys():
                        memory_wise_files[nearest_memory].append(new_info)
                    else:
                        memory_wise_files[nearest_memory] = []
                        memory_wise_files[nearest_memory].append(new_info)
            else:
                ood_files.append(image_name)


        # Now evaluate the end classifiers
        total_correct = 0
        total = 0

        for each_memory in memory_wise_files.keys():

            memory_dir = os.path.join(self.memory_dir, each_memory)

            fp = open(os.path.join(memory_dir, "logit_mapping.json"), "r")
            logit_mapping = json.load(fp)
            fp.close()

            current_classifier = end_classifier_model(model_dir = os.path.join(self.memory_dir, each_memory), \
            logit_mapping = logit_mapping)

            # current_classifier = feature_based_model(model_dir = os.path.join(self.memory_dir, each_memory), \
            # logit_mapping = logit_mapping)

            correct_count, _, _  = current_classifier.test_model(memory_wise_files[each_memory])
            total_correct += float(correct_count)
            total += float(len(memory_wise_files[each_memory]))

            print("For memory - ", each_memory, " correctness - ", float(correct_count) /float(len(memory_wise_files[each_memory])))


        print("Final test accuracy - ", total_correct / total)

    def evaluate_test_voting(self, memory_mapping_file, test_folder, class_list):
        fp = open(memory_mapping_file, "r")
        memory_distances = json.load(fp)
        fp.close()

        test_files = split_data.get_classwise_file_list(test_folder, class_list)
        memory_wise_files = {}
        file_wise_predictions = {}
        gnd_truth_dict = {}

        ood_files = []

        # Evaluate the memory models first, this is because otherwise the individual classifiers
        # have to be loaded on-off again, causing slowdown

        for each_item in test_files:
            image_name = each_item["image"]
            gnd_truth_dict[image_name] = each_item["class_name"]

            if image_name in memory_distances.keys():
                all_distances = memory_distances[image_name]["distances"]
                ground_truth_class_name = memory_distances[image_name]["orig_data"]["class_name"]

                # Right now it finds the nearest memory

                for each_memory in all_distances.keys():
                    memory_distance = all_distances[each_memory]

                    if each_memory in memory_wise_files.keys():
                        memory_wise_files[each_memory].append({"image":image_name, \
                        "mask" : each_item["mask"], "lesion_info" : each_item["lesion_info"],\
                        "class_name" : ground_truth_class_name})
                    else:
                        memory_wise_files[each_memory] = []
                        memory_wise_files[each_memory].append({"image":image_name, \
                        "mask" : each_item["mask"], "lesion_info" : each_item["lesion_info"],\
                        "class_name" : ground_truth_class_name})

            else:
                ood_files.append(image_name)

        # Now evaluate the end classifiers

        total_count = 0

        for each_memory in memory_wise_files.keys():

            memory_dir = os.path.join(self.memory_dir, each_memory)

            fp = open(os.path.join(memory_dir, "logit_mapping.json"), "r")
            logit_mapping = json.load(fp)
            fp.close()

            current_classifier = baseline_model(model_dir = os.path.join(self.memory_dir, each_memory), \
            logit_mapping = logit_mapping)

            correct_count, predictions, files_used = current_classifier.test_model(memory_wise_files[each_memory])

            assert len(memory_wise_files[each_memory]) >= len(files_used)
            for index in range(len(files_used)):

                filename = files_used[index]
                gnd_truth = gnd_truth_dict[filename]

                if filename not in file_wise_predictions.keys():
                    file_wise_predictions[filename] = [ [predictions[index], gnd_truth] ]
                else:
                    file_wise_predictions[filename].append([predictions[index], gnd_truth] )

            total_count += len(memory_wise_files[each_memory])

        total_correct = 0
        # Now perform voting -- right now takes all the votes into account
        for file in file_wise_predictions.keys():
            gnd_truth = file_wise_predictions[file][0][1]

            prediction_dict = {}
            for each_vote in file_wise_predictions[file]:
                # print("Each vote - ", each_vote)
                # print("prediction_dict - ", prediction_dict)
                if each_vote[0] not in prediction_dict.keys():
                    prediction_dict[each_vote[0]] = 1
                else:
                    prediction_dict[each_vote[0]] += 1
            max_pred = None
            max_vote = 0
            for each_pred in prediction_dict.keys():
                if prediction_dict[each_pred] > max_vote :
                    max_vote = prediction_dict[each_pred]
                    max_pred = each_pred

            if max_pred is not None and  (gnd_truth == max_pred ) :
                total_correct += 1


        print("Final test accuracy - ", float(total_correct) / float(total_count))


    def isolate_memory_data(self, memory_name, test_dir, class_list, memory_mapping_file, dest_dir, face_data_flag = False):


        # Find the training images for this memory, and copy the files under the train directory is dest
        training_data = parse_data(self.memory_dir)
        training_data_memory = training_data[memory_name]

        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        training_data_dir = os.path.join(dest_dir, "train_set")
        if not os.path.exists(training_data_dir):
            os.makedirs(training_data_dir)

        fp = open(os.path.join(training_data_dir, "logit_mapping.json"), "w")
        json.dump(training_data_memory["logit_mapping"], fp)
        fp.close()

        fp = open(os.path.join(training_data_dir, "weights.json"), "w")
        json.dump(training_data_memory["weight"], fp)
        fp.close()

        fp = open(os.path.join(training_data_dir, "samples.json"), "w")
        json.dump(training_data_memory["samples"], fp)
        fp.close()

        for each_sample in training_data_memory["samples"]:
            class_name = each_sample["class_name"]
            image_source = each_sample["image"]

            pieces = image_source.split("/")
            image_name = pieces[-1]

            class_dir = os.path.join(training_data_dir, class_name)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

            copyfile(image_source, os.path.join(class_dir, image_name))



        fp = open(memory_mapping_file, "r")
        memory_distances = json.load(fp)
        fp.close()

        test_files = split_data.read_training_data(test_dir)
        memory_wise_files = {}
        memory_wise_files[memory_name] = []


        test_set_dir = os.path.join(dest_dir, "test_set")
        if not os.path.exists(test_set_dir):
            os.makedirs(test_set_dir)

        # Find the test images which are getting mapped to this memory, and copy them under


        for each_item in test_files:
            image_name = each_item["image"]

            if image_name not in memory_distances.keys():
                continue

            all_distances = memory_distances[image_name]["distances"]
            ground_truth_class_name = memory_distances[image_name]["orig_data"]["class_name"]

            nearest_memory = None
            closest_distance = np.inf

            # If current memory is in the list of admitted memories for this file.

            if memory_name in memory_distances[image_name]["distances"].keys():

                memory_wise_files[memory_name].append({"image":image_name, \
                "class_name" : ground_truth_class_name})


                class_name = each_item["class_name"]
                image_source = each_item["image"]

                pieces = image_source.split("/")
                image_name = pieces[-1]

                class_dir = os.path.join(test_set_dir, class_name)
                if not os.path.exists(class_dir):
                    os.makedirs(class_dir)

                copyfile(image_source, os.path.join(class_dir, image_name))



        test_data_file = os.path.join(dest_dir, "test_set.json")
        fp = open(test_data_file, "w")
        json.dump(memory_wise_files[memory_name], fp)
        fp.close()


    def train_baseline(self, class_list, dest_dir):


        self.training_data = split_data.get_classwise_file_list(dest_dir, class_list)
        self.training_epochs = 40
        dest_dir = os.path.join(dest_dir, "baseline_model")
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        # Modify training data to make it look proper.
        self.training_data = convert_representation(self.training_data, weighted = True)

        current_classifier = end_classifier_model(model_dir =  dest_dir, \
        logit_mapping = self.training_data["logit_mapping"],\
        weights =  list(self.training_data["weight"].values()))

        training_accuracy = current_classifier.train_model(self.training_data["samples"], self.training_epochs)

        # Saving the logit to class names file

        fp = open(os.path.join(dest_dir, "logit_mapping.json"), "w")
        json.dump(self.training_data["logit_mapping"], fp)
        fp.close()

    def test_baseline(self, class_list, test_dir, trained_model_dir):

        test_files = split_data.get_classwise_file_list(test_dir, class_list)

        dest_dir = os.path.join(trained_model_dir, "baseline_model")
        assert os.path.exists(dest_dir)

        fp = open(os.path.join(dest_dir, "logit_mapping.json"), "r")
        logit_mapping = json.load(fp)
        fp.close()

        current_classifier = end_classifier_model(dest_dir, logit_mapping)

        _, _, _  = current_classifier.test_model(test_files)

    def train_baseline_model(self, class_list, dest_dir):
        self.training_data = parse_data(self.memory_dir, weighted = False)

        memory_dir = os.path.join(self.memory_dir, each_memory)

        current_classifier = end_classifier_model(model_dir =  memory_dir, \
        logit_mapping = self.training_data[each_memory]["logit_mapping"],\
        weights =  list(self.training_data[each_memory]["weight"].values()))

        training_accuracy = current_classifier.train_model(self.training_data[each_memory]["samples"], self.training_epochs)

        # Saving the logit to class names file
        fp = open(os.path.join(memory_dir, "logit_mapping.json"), "w")
        json.dump(self.training_data[each_memory]["logit_mapping"], fp)
        fp.close()