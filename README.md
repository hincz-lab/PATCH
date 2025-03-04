# PATCH
Code and sample data from development of the PATCH (Pairwise Assignment Training for Classifying Heterogeneity) machine learning method

The code in this repository is from Van Horn, et al., PATCH: a deep learning method to assess heterogeneity of artistic practice in historical paintings. Currently a preprint on Arxiv 
(https://doi.org/10.48550/arXiv.2502.01912)

The data are topographic images of paintings from Ji, et al., 2021

To run patch, run "PATCH_Van-Horn-et-al_0-0_040325av1253.py" in the terminal as follows:

        python PATCH_Van-Horn-et-al_0-0_040325av1253.py > my_out_file.txt

The out file records tensorflow output for each epoch of each fold, including validation accuracy (our metric of interest).
Once analysis is complete, use run "PATCH_outputparser_0-0_040325av1531.py" in the terminal to output mean val accuracy, max val accuracy, and number of patches to use 
for comparison to random assignment distribution (RAD). 

Use "PATCH_RAD_0-0_040325av1555.ipynb" to calculate the mean, standard deviation, and hypothetical maximum value for RAD for the given number of patches (number in the smaller image)
Use the spreadsheet program of your choice to compare your observed disribution to the RAD.
