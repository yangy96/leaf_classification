import model
import sys
import parse_data

train_dir = "/Users/souradeep/Documents/Datasets/Acne/ACNE/training_data/"
# test_dir = "/Users/souradeep/Documents/Datasets/Acne/ACNE/training_data_rotated/"
test_dir = "/Users/souradeep/Documents/Datasets/Acne/ACNE/training_data_blurred/"

# train_dir = "/data5/yangy96/acne_data/lesion_images_split/train/"
# test_dir = "/data5/yangy96/acne_data/lesion_images_split/test/"


class_name_dictionary = {"15" : 0, "20" : 1, "35" : 2}
# class_name_dictionary = {"20" : 0, "35" : 1}

training_data = parse_data.parse_data(train_dir, class_name_dictionary)
current_model = model.baseline_model(class_name_dictionary)
# avg_error = current_model.train_model(training_data, 20)

test_data = parse_data.parse_data(test_dir, class_name_dictionary)

current_model.test_model(test_data)
