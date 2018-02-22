#!/usr/bin/env python3
"""
Author: David Crook
Copyright Microsoft Corporation 2017
"""
import PreProcess
import pandas as pd

#location to drop the processed label file used for creating TF Records
#base directory to drop tf records for training
TF_REC_DEST = 'DL/tfrecords/'
#Where the current CIFAR 10 labels file exists
labels_file = 'morphLabels.csv'
#Where the images are stored
base_imgs_dir = 'morph/images/'
#Where to drop the dictionary for converting node index to label string
conversion_dict_path = './proc_train_classes_dictionary.csv'

def create_fqp_lbl_file():
    labels = pd.read_csv(labels_file, dtype={'id': object})
    labels['id'] = base_imgs_dir + labels['id'] + '.png'
    classes = list(set(labels['label']))
    num_class = [x for x in range(len(classes))]
    class_to_num_dict = dict(zip(classes, num_class))
    proc_labels = labels.replace({'label': class_to_num_dict})
    proc_labels.columns = ['examplepath', 'label']
    proc_labels.to_csv(LABELS_FILE, index = False)
    conv_df = pd.DataFrame(list(class_to_num_dict.items()))
    conv_df.columns = ['class', 'node']
    conv_df.to_csv(conversion_dict_path, index = False)

def main():
    '''
    Main function which converts a label file into tf records
    '''
    create_fqp_lbl_file()
    PreProcess.write_records_from_file(LABELS_FILE, TF_REC_DEST, 4)

if __name__ == "__main__":
    main()
