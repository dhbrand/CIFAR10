#!/usr/bin/env python3
"""
Author: David Crook
Copyright Microsoft Corporation 2017
"""
import PreProcess

LABELS_FILE = 'C:/data/cifar_10/proc_train_labels.csv'
TF_REC_DEST = 'C:/data/cifar_10/tfrecords/'

def main():
    '''
    Main function which converts a label file into tf records
    '''
    PreProcess.write_records_from_file(LABELS_FILE, TF_REC_DEST, 4)

if __name__ == "__main__":
    main()
