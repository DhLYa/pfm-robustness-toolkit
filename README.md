# pfm-robustness-toolkit

This project is based on the original reetoolbox project https://github.com/alexjfoote/reetoolbox,
which is licensed under the Apache License 2.0.

This repository contains modifications and additions by Dhyey Yajnik.

## Status

This repo contains the initial project files.
Ongoing changes will focus on improving usability
and easier configuration for new datasets/models for use in future robustness experiments

## Description
The Robustness Evaluation and Enhancement Toolbox (REEToolbox or REET: paper available at https://academic.oup.com/bioinformatics/article-abstract/38/12/3312/6582557) provides tools for measuring and improving the robustness of ML models. REEToolbox uses adversarial transforms - data transforms that are adversarially optimised to fool a model - to generate challenging transformations of input data. For example, in the image below a transform that simulates changing the staining of a tissue image has been optimised to cause a trained model to misclassify a patch of tumorous tissue as non-tumorous. 

