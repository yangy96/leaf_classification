import os
import random
import json

def parse_data(source_dir, weighted = False):

    memory_suffix = "_memory"
    content_filename = "training_data.json"

    training_data = {}
    logit_mapping = {}

    for root, dirs, files in os.walk(source_dir):
        if any(dirs):
            for each_mem in dirs:
                if each_mem.endswith(memory_suffix):
                    memory_dir = os.path.join(root, each_mem)
                    training_images_file = os.path.join(memory_dir, content_filename)

                    fp = open(training_images_file, "r")
                    samples_list = json.load(fp)
                    fp.close()

                    weight = {}
                    logit_mapping = {}
                    for each_line in samples_list :

                        if each_line["class_name"] not in logit_mapping.keys() :
                            logit_mapping[each_line["class_name"]] = len(logit_mapping)

                        if logit_mapping[each_line["class_name"]] in weight.keys():
                            weight[logit_mapping[each_line["class_name"]]] += 1
                        else:
                            weight[logit_mapping[each_line["class_name"]]] = 1


                    for each_logit in weight.keys():
                        # For ACNE lesion : no balancing in node classifiers worked better
                        if weighted :
                            weight[each_logit] = (float(len(samples_list))) / float(weight[each_logit])
                        else:
                            weight[each_logit] = 1.0

                    random.shuffle(samples_list)
                    training_data[each_mem] = {"samples" : samples_list, "logit_mapping" : logit_mapping, "weight" : weight}

    return training_data


def convert_representation(samples_list, weighted = False):

    weight = {}
    logit_mapping = {}
    for each_line in samples_list :

        if each_line["class_name"] not in logit_mapping.keys() :
            logit_mapping[each_line["class_name"]] = len(logit_mapping)

        if logit_mapping[each_line["class_name"]] in weight.keys():
            weight[logit_mapping[each_line["class_name"]]] += 1
        else:
            weight[logit_mapping[each_line["class_name"]]] = 1


    for each_logit in weight.keys():
        # For ACNE lesion : no balancing in node classifiers worked better
        if weighted :
            weight[each_logit] = (float(len(samples_list))) / float(weight[each_logit])
        else:
            weight[each_logit] = 1.0

    random.shuffle(samples_list)
    new_rep = {"samples" : samples_list, "logit_mapping" : logit_mapping, "weight" : weight}

    return new_rep
