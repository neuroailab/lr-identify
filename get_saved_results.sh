#!/bin/bash

mkdir -p ./saved_classifier_results/
curl -fLo ./saved_classifier_results/randomforest_allobs_results.pkl https://lridentify-savedcls.s3-us-west-1.amazonaws.com/randomforest_allobs_results.pkl
curl -fLo ./saved_classifier_results/svm_allobs_results.pkl https://lridentify-savedcls.s3-us-west-1.amazonaws.com/svm_allobs_results.pkl
curl -fLo ./saved_classifier_results/conv1dmlp_allobs_results.pkl https://lridentify-savedcls.s3-us-west-1.amazonaws.com/conv1dmlp_allobs_results.pkl
