import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
import torchvision
from torchvision import datasets, models, transforms, utils
from PIL import Image
import sys
import os
import json
import pathlib
import cv2
from distance_calculations.find_features import return_feature_vector
from joblib import dump, load
from end_classifiers.net_models import shallow, resnet34, resnet50, resnet101, alexnet, googlenet, vgg16, mobilenet_v2

verbosity = True

def adjust_gamma(PIL_image, gamma=1.0):

    # build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
    opencvImage = cv2.cvtColor(np.array(PIL_image), cv2.COLOR_RGB2BGR)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table

    gamma_altered_image = cv2.LUT(opencvImage, table)

    gamma_altered_image = cv2.cvtColor(gamma_altered_image, cv2.COLOR_BGR2RGB)
    ret_pil_image = Image.fromarray(gamma_altered_image)

    return ret_pil_image

def pre_process(filename):
    ''' Takes in a normal file, and produces a PIL image'''
    ''' We might be interested in doing things like segmentation etc before feeding in '''

    input_image = Image.open(filename)
    return input_image

def read_features(filename):
    ''' Takes in a normal file, and produces a PIL image'''
    ''' We might be interested in doing things like segmentation etc before feeding in '''

    fp = open(filename, "r")
    features = json.load(fp)
    fp.close()
    features = features['training_feature']
    predicted_label = features['predicted_label']

    one_hot_rep = {}
    one_hot_rep["black_flag"] = features["black_flag"]

    if predicted_label == "1015" :
        one_hot_rep["1015"] = 1.0
    else:
        one_hot_rep["1015"] = 0.0

    if predicted_label == "20" :
        one_hot_rep["20"] = 1.0
    else:
        one_hot_rep["20"] = 0.0

    if predicted_label == "303536" :
        one_hot_rep["303536"] = 1.0
    else:
        one_hot_rep["303536"] = 0.0

    feat_list = [one_hot_rep["black_flag"], one_hot_rep["1015"], one_hot_rep["20"], one_hot_rep["303536"]]

    torch_features = torch.tensor(feat_list, requires_grad = False, dtype = torch.float32)

    return torch_features

class Net(nn.Module):
    def __init__(self, no_of_outputs):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(44944, 2000)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(2000, 100)
        self.fc3 = nn.Linear(100, no_of_outputs)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class augmented_Resnet(nn.Module):
    def __init__(self, no_of_features, no_of_outputs, device):
        super().__init__()

        # HARD-CODING-ALERT
        resnet_features_shape = 2048
        self.resnet_model = models.resnet50(pretrained = True)
        self.resnet_model.fc = Identity()

        for param in self.resnet_model.parameters():
            param.requires_grad = False

        self.linear_layer = nn.Linear(resnet_features_shape + no_of_features, no_of_outputs)

        self.resnet_model.to(device)
        self.linear_layer.to(device)

    def forward(self, input_image_batch, input_feature_batch, train = True):

        if train:
            image_features = self.resnet_model(input_image_batch)
            augmented_features = torch.cat((image_features, input_feature_batch), 1)

            output = self.linear_layer(augmented_features)
        else:
            with torch.no_grad():
                image_features = self.pre_resnet_model(input_image_batch)
                augmented_features = torch.cat((image_features, input_feature_batch), 0)
                output = self.linear_layer(augmented_features)

        probabilities = torch.nn.functional.softmax(output, dim = 1)
        return probabilities

        return x


class end_classifier_model():

    def __init__(self, model_dir, logit_mapping, weights = []):
        self.model_dir = model_dir
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.batch_size = 64
        # batch size 64 worked wonders for ACNE lesion classification
        self.logit_mapping = logit_mapping
        self.output_count = len(logit_mapping)

        self.model = self.model_3(self.output_count)
        self.model_filename = os.path.join(self.model_dir, "local_model.pth")

        if len(weights) != 0 :
            self.criterion = nn.CrossEntropyLoss(weight = torch.tensor(weights, requires_grad = False, device = self.device))
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Lesion classification params --
        self.optimizer = optim.SGD(self.pre_trained_model.parameters(), lr = 0.01, momentum = 0.8) # 0.8
        self.exp_lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)

        # self.network_preprocess = transforms.Compose([transforms.ToTensor(),])
        self.network_preprocess = transforms.Compose([
                                                     transforms.Resize(256),
                                                     transforms.CenterCrop(224),
                                                     transforms.ToTensor(),
                                                     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                     ])

    def model_1(self, no_of_outputs):

        self.pre_trained_model = models.resnet50(pretrained = True)
        for param in self.pre_trained_model.parameters():
            param.requires_grad = False

        # For Resnet
        self.pre_trained_model.fc = nn.Linear(self.pre_trained_model.fc.in_features, no_of_outputs)

        # For Densenet
        # self.pre_trained_model.classifier = nn.Linear(1024, no_of_outputs)

        # For VGG net
        # self.pre_trained_model.classifier[6] = nn.Linear(4096, no_of_outputs)

        self.pre_trained_model.to(self.device)

        return self.pre_trained_model

    def model_2(self, no_of_outputs):

        self.pre_trained_model = Net(no_of_outputs)
        self.pre_trained_model.to(self.device)

        return self.pre_trained_model
    
    def model_3(self,no_of_outputs):
        
        self.pre_trained_model = resnet50(pretrained=True, num_classes=no_of_outputs)
        self.pre_trained_model.to(self.device)
        """if model_name == 'resnet34':
            model = resnet34(pretrained=pretrained, num_classes=num_classes)
        
        if model_name == 'resnet101':
            model = resnet101(pretrained=pretrained, num_classes=num_classes)
        
        if model_name == 'alexnet':
            model = alexnet(pretrained=pretrained, num_classes=num_classes)
            
        if model_name == 'googlenet':
            model = googlenet(pretrained=pretrained, num_classes=num_classes)
        
        if model_name == 'vgg16':
            model = vgg16(pretrained=pretrained, num_classes=num_classes)
            
        if model_name == 'mobilenet_v2':
            model = mobilenet_v2(pretrained=pretrained, num_classes=num_classes)"""

        return self.pre_trained_model

    def forward_compute(self, input_batch ,train = True):

        if train:
            output = self.model(input_batch)
        else:
            with torch.no_grad():
                output = self.model(input_batch)

        probabilities = torch.nn.functional.softmax(output, dim = 1)
        return probabilities

    def train_model(self, filenames_and_label, epochs = 100):

        assert len(filenames_and_label) > 0

        for index in range(epochs):

            total_error = 0
            start_index = 0
            end_index = min(self.batch_size, len(filenames_and_label))

            while start_index < len(filenames_and_label) :

                end_index = start_index + self.batch_size
                if end_index > len(filenames_and_label):
                    end_index = len(filenames_and_label)

                self.optimizer.zero_grad()
                inputs, labels, files_involved = self.build_data_set(filenames_and_label[start_index:end_index])

                predicted_labels = self.forward_compute(inputs)
                loss = self.criterion(predicted_labels, labels)
                loss.backward()
                self.optimizer.step()

                # print("For range - (", start_index, ", ", end_index, ") loss computed - ", loss)

                start_index += self.batch_size


                for each_item in zip(labels.tolist(), predicted_labels):
                    pred = torch.argmax(each_item[1]).tolist()
                    if each_item[0] != pred:
                        total_error += 1

            print("At episode - ", index, " error - ", float(total_error)/float(len(filenames_and_label)))

            self.exp_lr_scheduler.step()
            torch.save(self.model.state_dict(), self.model_filename)
        print("Error at end - ", total_error/float(len(filenames_and_label)))

    def build_data_set(self, filenames_and_label):

        tensor_collection = []
        label_collection = []
        image_filenames = []

        for each_item in filenames_and_label:

            if each_item["class_name"] in self.logit_mapping.keys():
                PIL_image = pre_process(each_item["image"])
                image_filenames.append(each_item["image"])
                input_tensor = self.network_preprocess(PIL_image)
                tensor_collection.append(input_tensor)
                label_collection.append(torch.tensor(self.logit_mapping[each_item["class_name"]], requires_grad = False))
            else:
                print("Ignoring test file")

        if len(tensor_collection) != 0:
            tensor_collection = torch.stack(tensor_collection, dim = 0)
            label_collection = torch.stack(label_collection, dim = 0)
            tensor_collection = tensor_collection.to(self.device)
            label_collection = label_collection.to(self.device)
        else:
            tensor_collection = None
            label_collection = None

        return tensor_collection, label_collection, image_filenames

    def build_data_set_with_augmentation(self, filenames_and_label):

        tensor_collection = []
        label_collection = []
        image_filenames = []

        for each_item in filenames_and_label:

            if each_item["class_name"] in self.logit_mapping.keys():

                for angle in [0, 90, 180, 270]:
                    PIL_image = pre_process(each_item["image"])
                    PIL_image = PIL_image.rotate(angle)

                    for g_index in range(4):
                        gamma = 1.0 + float(g_index) * (1.0/4.0)
                        PIL_image = adjust_gamma(PIL_image, gamma)
                        image_filenames.append(each_item["image"])
                        input_tensor = self.network_preprocess(PIL_image)
                        tensor_collection.append(input_tensor)
                        label_collection.append(torch.tensor(self.logit_mapping[each_item["class_name"]], requires_grad = False))
            else:
                print("Ignoring test file")

        if len(tensor_collection) != 0:
            tensor_collection = torch.stack(tensor_collection, dim = 0)
            label_collection = torch.stack(label_collection, dim = 0)
            tensor_collection = tensor_collection.to(self.device)
            label_collection = label_collection.to(self.device)
        else:
            tensor_collection = None
            label_collection = None

        return tensor_collection, label_collection, image_filenames

    def test_model(self, filenames_and_label):

        file = pathlib.Path(self.model_filename)
        if file.exists():
            self.model.load_state_dict(torch.load(self.model_filename, map_location = self.device))
            self.model.eval()
        else:
            print("Model file being searched for during testing does not exist.")
            return None

        assert len(filenames_and_label) > 0

        # Compute the error here
        class_name_list = list(self.logit_mapping.keys())
        class_index_list = list(self.logit_mapping.values())


        correct_count = 0.0
        total_count = 0.0
        total_error = 0
        start_index = 0
        end_index = 0
        prediction_labels = []
        files_involved = []
        while start_index < len(filenames_and_label) :


            end_index = start_index + self.batch_size
            if end_index > len(filenames_and_label):
                end_index = len(filenames_and_label)

            inputs, ground_truth_labels, files_used = self.build_data_set(filenames_and_label[start_index:end_index])

            if inputs is not None:

                files_involved += files_used


                predicted_labels = self.forward_compute(inputs, False)
                py_labels = predicted_labels.tolist()
                for each_label in py_labels:

                    current_label = each_label.index(max(each_label))

                    position = class_index_list.index(current_label)
                    class_name = class_name_list[position]
                    prediction_labels.append(class_name)


                for each_item in zip(ground_truth_labels.tolist(), predicted_labels):
                    pred = torch.argmax(each_item[1]).tolist()
                    # print("Grnd - ", each_item[0], " pred - ", pred)
                    if each_item[0] == pred:
                        correct_count += 1.0
                    total_count += 1.0

            start_index += self.batch_size

        if total_count != 0:
            print("Test accurcy - ", (correct_count/total_count) * 100.0, "%", " correct - ", correct_count,  " total - ", total_count)

        return correct_count, prediction_labels, files_involved


class feature_augmented_model():

    def __init__(self, model_dir, logit_mapping, weights = []):
        self.model_dir = model_dir
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.batch_size = 64
        # batch size 64 worked wonders for ACNE lesion classification
        self.logit_mapping = logit_mapping
        self.output_count = len(logit_mapping)

        # HARD-CODING-ALERT
        self.model = augmented_Resnet(4, self.output_count, self.device)
        self.model_filename = os.path.join(self.model_dir, "local_model.pth")

        if len(weights) != 0 :
            self.criterion = nn.CrossEntropyLoss(weight = torch.tensor(weights, requires_grad = False, device = self.device))
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Lesion classification params --
        self.optimizer = optim.SGD(self.model.parameters(), lr = 0.01, momentum = 0.9) # 0.8
        self.exp_lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)

        # self.network_preprocess = transforms.Compose([transforms.ToTensor(),])
        self.network_preprocess = transforms.Compose([
                                                     transforms.Resize(256),
                                                     transforms.CenterCrop(224),
                                                     transforms.ToTensor(),
                                                     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                     ])

    def model_1(self, no_of_features, no_of_outputs):

        self.pre_trained_model = models.resnet50(pretrained = True)
        for param in self.pre_trained_model.parameters():
            param.requires_grad = False

        # For Resnet
        self.linear_layer = nn.Linear(self.pre_trained_model.fc.in_features + no_of_features, no_of_outputs)


        # For Densenet
        # self.pre_trained_model.classifier = nn.Linear(1024, no_of_outputs)

        # For VGG net
        # self.pre_trained_model.classifier[6] = nn.Linear(4096, no_of_outputs)

        self.pre_trained_model.to(self.device)
        self.linear_layer.to(self.device)

        return self.pre_trained_model

    def model_2(self, no_of_outputs):

        self.pre_trained_model = Net(no_of_outputs)
        self.pre_trained_model.to(self.device)

        return self.pre_trained_model

    def forward_compute(self, input_image_batch, input_feature_batch, train = True):

        return self.model.forward(input_image_batch, input_feature_batch)

    def train_model(self, filenames_and_label, epochs = 100):

        assert len(filenames_and_label) > 0

        for index in range(epochs):

            total_error = 0
            start_index = 0
            end_index = min(self.batch_size, len(filenames_and_label))

            while start_index < len(filenames_and_label) :

                end_index = start_index + self.batch_size
                if end_index > len(filenames_and_label):
                    end_index = len(filenames_and_label)

                self.optimizer.zero_grad()
                image_inputs, feature_inputs, labels, files_involved = self.build_data_set(filenames_and_label[start_index:end_index])

                predicted_labels = self.forward_compute(image_inputs, feature_inputs)
                loss = self.criterion(predicted_labels, labels)
                loss.backward()
                self.optimizer.step()

                # print("For range - (", start_index, ", ", end_index, ") loss computed - ", loss)

                start_index += self.batch_size


                for each_item in zip(labels.tolist(), predicted_labels):
                    pred = torch.argmax(each_item[1]).tolist()
                    if each_item[0] != pred:
                        total_error += 1

            print("At episode - ", index, " error - ", float(total_error)/float(len(filenames_and_label)))

            self.exp_lr_scheduler.step()
            torch.save(self.model.state_dict(), self.model_filename)
        print("Error at end - ", total_error/float(len(filenames_and_label)))

    def build_data_set(self, filenames_and_label):

        feature_collection = []
        tensor_collection = []
        label_collection = []
        image_filenames = []

        for each_item in filenames_and_label:

            if each_item["class_name"] in self.logit_mapping.keys():
                PIL_image = pre_process(each_item["image"])
                image_filenames.append(each_item["image"])
                image_tensor = self.network_preprocess(PIL_image)
                im_features = read_features(each_item["lesion_info"])

                feature_collection.append(im_features)
                tensor_collection.append(image_tensor)
                label_collection.append(torch.tensor(self.logit_mapping[each_item["class_name"]], requires_grad = False))
            else:
                print("Ignoring test file")

        if len(tensor_collection) != 0:
            tensor_collection = torch.stack(tensor_collection, dim = 0)
            feature_collection = torch.stack(feature_collection, dim = 0)
            label_collection = torch.stack(label_collection, dim = 0)
            tensor_collection = tensor_collection.to(self.device)
            label_collection = label_collection.to(self.device)
            feature_collection = feature_collection.to(self.device)
        else:
            tensor_collection = None
            label_collection = None
            feature_collection = None

        return tensor_collection, feature_collection, label_collection, image_filenames


    def test_model(self, filenames_and_label):

        file = pathlib.Path(self.model_filename)
        if file.exists():
            self.model.load_state_dict(torch.load(self.model_filename, map_location = self.device))
            self.model.eval()
        else:
            print("Model file being searched for during testing does not exist.")
            return None

        assert len(filenames_and_label) > 0

        # Compute the error here
        class_name_list = list(self.logit_mapping.keys())
        class_index_list = list(self.logit_mapping.values())


        correct_count = 0.0
        total_count = 0.0
        total_error = 0
        start_index = 0
        end_index = 0
        prediction_labels = []
        files_involved = []
        while start_index < len(filenames_and_label) :


            end_index = start_index + self.batch_size
            if end_index > len(filenames_and_label):
                end_index = len(filenames_and_label)

            image_inputs, feature_inputs, ground_truth_labels, files_used = self.build_data_set(filenames_and_label[start_index:end_index])


            if image_inputs is not None:

                files_involved += files_used

                predicted_labels = self.forward_compute(image_inputs, feature_inputs, False)
                py_labels = predicted_labels.tolist()
                for each_label in py_labels:

                    current_label = each_label.index(max(each_label))

                    position = class_index_list.index(current_label)
                    class_name = class_name_list[position]
                    prediction_labels.append(class_name)


                for each_item in zip(ground_truth_labels.tolist(), predicted_labels):
                    pred = torch.argmax(each_item[1]).tolist()
                    # print("Grnd - ", each_item[0], " pred - ", pred)
                    if each_item[0] == pred:
                        correct_count += 1.0
                    total_count += 1.0

            start_index += self.batch_size

        if total_count != 0:
            print("Test accurcy - ", (correct_count/total_count) * 100.0, "%", " correct - ", correct_count,  " total - ", total_count)

        return correct_count, prediction_labels, files_involved

class SVM_classifier():
    # TRY THIS
    # Take the code from here, and give it a shot : https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

    def __init__(self, model_dir, logit_mapping, weights = []):
        self.model_dir = model_dir
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        self.logit_mapping = logit_mapping
        self.output_count = len(logit_mapping)
        self.batch_size = 32
        self.features_count = 1000
        self.weights = weights
        self.use_features_only = False # For True, turn features_count to 2

        self.model_filename = os.path.join(self.model_dir, "SVM_model.pth")

        self.network_preprocess = transforms.Compose([
                                                     transforms.Resize(256),
                                                     transforms.CenterCrop(224),
                                                     transforms.ToTensor(),
                                                     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                     ])

    def compute_features(self, input_batch):




        self.pre_trained_model = models.resnet18(pretrained = True)
        for param in self.pre_trained_model.parameters():
            param.requires_grad = False


        with torch.no_grad():
            output = self.pre_trained_model(input_batch)
            self.features = torch.nn.functional.softmax(output, dim = 1)
            # self.features = output
        self.features = self.features.cpu().detach().numpy()

        return self.features



    def train_model(self, filenames_and_label, epochs = 100):

        if self.output_count == 1:
            return

        assert len(filenames_and_label) > 0

        total_error = 0
        start_index = 0
        end_index = min(self.batch_size, len(filenames_and_label))

        image_features = np.empty((0, self.features_count), float)
        labels = np.empty((0), float)

        while start_index < len(filenames_and_label) :

            end_index = start_index + self.batch_size
            if end_index > len(filenames_and_label):
                end_index = len(filenames_and_label)

            if self.use_features_only :
                network_features, current_labels, current_files_involved = self.build_data_set_features(filenames_and_label[start_index:end_index])
            else:
                current_image_inputs, current_labels, current_files_involved = self.build_data_set(filenames_and_label[start_index:end_index])
                network_features = self.compute_features(current_image_inputs)

            image_features = np.append(image_features, network_features, axis = 0)

            current_labels = current_labels.cpu().detach().numpy()
            labels = np.append(labels, current_labels, axis = 0)

            print("For range - (", start_index, ", ", end_index, ") computed features.")
            start_index += self.batch_size

        print("Setting out to train SVM .. ")

        clf = make_pipeline(StandardScaler(), SVC(gamma='auto' ))

        clf.fit(image_features, labels)

        print("SVM trained . ")
        dump(clf, self.model_filename)

        correct = 0.0
        total = 0.0
        index = 0
        for each_feature in image_features:
            pred = clf.predict([each_feature])
            if pred == labels[index] :
                correct += 1.0

            index += 1
            total += 1.0

        print("Training accuracy - ", float(correct) / float(total))


    def build_data_set(self, filenames_and_label):

        tensor_collection = []
        label_collection = []
        image_filenames = []

        for each_item in filenames_and_label:

            if each_item["class_name"] in self.logit_mapping.keys():
                PIL_image = pre_process(each_item["image"])
                image_filenames.append(each_item["image"])
                input_tensor = self.network_preprocess(PIL_image)
                tensor_collection.append(input_tensor)
                label_collection.append(torch.tensor(self.logit_mapping[each_item["class_name"]], requires_grad = False))
            else:
                print("Ignoring test file")

        if len(tensor_collection) != 0:
            tensor_collection = torch.stack(tensor_collection, dim = 0)
            label_collection = torch.stack(label_collection, dim = 0)
            tensor_collection = tensor_collection.to(self.device)
            label_collection = label_collection.to(self.device)
        else:
            tensor_collection = None
            label_collection = None

        return tensor_collection, label_collection, image_filenames

    def build_data_set_features(self, filenames_and_label):

        feature_collection = []
        label_collection = []
        image_filenames = []

        for each_item in filenames_and_label:

            if each_item["class_name"] in self.logit_mapping.keys():
                im_features = np.array(list(each_item["features"].values()))
                feature_collection.append(im_features)
                label_collection.append(torch.tensor(self.logit_mapping[each_item["class_name"]], requires_grad = False))
            else:
                print("Ignoring test file")

        feature_collection = np.array(feature_collection)
        if len(feature_collection) != 0:
            label_collection = torch.stack(label_collection, dim = 0)
            label_collection = label_collection.to(self.device)
        else:
            label_collection = None

        return feature_collection, label_collection, image_filenames


    def test_model(self, filenames_and_label):

        assert len(filenames_and_label) > 0

        # Compute the error here
        class_name_list = list(self.logit_mapping.keys())
        class_index_list = list(self.logit_mapping.values())

        clf = load(self.model_filename)


        correct_count = 0.0
        total_count = 0.0
        total_error = 0
        start_index = 0
        end_index = 0
        prediction_labels = []
        files_involved = []



        while start_index < len(filenames_and_label) :


            end_index = start_index + self.batch_size
            if end_index > len(filenames_and_label):
                end_index = len(filenames_and_label)

            image_features = np.empty((0, self.features_count), float)
            labels = np.empty((0), float)

            if self.use_features_only :
                network_features, current_labels, current_files_involved = self.build_data_set_features(filenames_and_label[start_index:end_index])
            else:
                current_image_inputs, current_labels, current_files_involved = self.build_data_set(filenames_and_label[start_index:end_index])
                network_features = self.compute_features(current_image_inputs)

            image_features = np.append(image_features, network_features, axis = 0)
            current_labels = current_labels.cpu().detach().numpy()
            gnd_labels = np.append(labels, current_labels, axis = 0)


            if image_features is not None:

                files_involved += current_files_involved

                index = 0
                for each_label in gnd_labels:

                    if self.output_count > 1 :
                        prediction_index = clf.predict([image_features[index]])
                        position = class_index_list.index(prediction_index)
                        class_name = class_name_list[position]
                        prediction = int(class_name)
                    else:
                        prediction = int(list(self.logit_mapping.keys())[0])

                    ground_truth = gnd_labels[index]

                    if ground_truth == prediction:
                        correct_count += 1.0
                    total_count += 1.0
                    index += 1

            start_index += self.batch_size

        if total_count != 0:
            print("Test accurcy - ", (correct_count/total_count) * 100.0, "%", " correct - ", correct_count,  " total - ", total_count)

        return correct_count, prediction_labels, files_involved
