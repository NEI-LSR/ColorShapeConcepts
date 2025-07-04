# ColorShapeConcepts

Code needed for the paper "The representation of object concepts across the brain". Includes:
- Code for running searchlight cross decoding and other algorithms
- Generating rough versions of figures in the paper

Much of this code is dependent on our `neurotools` library ([github](https://github.com/spencer-loggia/neurotools)), which contains core algorithms and utility functions. 

## File Descriptions:
- analyze
  - `searchlight_runner.py`: contructs the LSDM (or standard searchlight), connects the LSDM to the fMRI dataloader that feed trial data and labels, runs cross validation procedure (identity and cross decoding), saves results to csv. Configuration can be changed at the top of the file. Takes ~8 hours to run on a GTX 4090TI GPU.
  - `simulation_runner.py`: contrusts an LSDM and standard searchlight, connects to the simulation dataloader, for different input set sizes preforms cross validated indentity and cross decoding of the simulated data for both models and saves the results to csv.
  - `compute_embeddings.py`: Given a trained LSDM output directory (containing binaries of trained models on each CV fold), computes the representational dissimilarity matrices for each ROI in the LSDM latent space (see paper methods) and saves them as npy files with a cssv key.
  - `analysisRL.m`: fits reinforcement learning models to behavioral data (long-term memory trials done on touchscreen tablets), and saves fits as .mat files and RL parameters in a csv
  - `bin_compute_confidence.py`: computes accuracies and confidence intervals for the touchscreen tablet behavioral data, and saves a series of numpy arrays in a compressed format, .npz
  - `learning_performance.py`: computes accuracies and confidence intervals for initial performance or plateau performance for different trial types for tasks of interest, saves out as csvs
  - `compute_selectivity.py`: computed color-associated vs non-color-associated shape selectivity and incongruent vs congruent object selectivity in ROI parcels, saves result in csv
  - `surface_comparison.py`: computes z-scored color-associated vs non-color-associated shape and incongruent vs congruent object contrasts in bins along the surface of each hemisphere, saves results in csvs
  - `color_biased_regions.py`: computes color-associated shape bias in color-biased vs. non-color-biased regions in V4 and IT parcels, saves results in csv
- bin
  - `dataloader.py` defines a class that is designed to load fMRI data in parallel from the data directory and serve batches of it to the searchlight runner.
  - `searchlight_dataloader.py` same as above, but generates simulated data as described in the methods.
  - `plotter.py` defines scripts used for generating plots of data.
  - `passive_task_functions.py` defines functions used in the `compute_selectivity.py` and `color_biased_regions.py`
- visualize
  - `decoding_plots_from_results.py` given the path(s) to output model diretory(s) (as producedby searchlight runner and containing 'results.csv') bootstrap the results, produce plots of decdoing performance over ROIs
  - `geometry_plots_from_results.py` given the path(s) to output model diretory(s) that have been processed by the `compute_embeddings.py` script and thus contain 'rdm_key.csv', produce MDS plots and bootstrapped plots of correlation with color space geometry.
  - `create_searchlight_simulation_plots.py` given the results of the `simulation_runner.py` script, create the simulation decoding plots shown in Fig. S3.
  - `decoding_results_to_surface.py` script to project the identity, cross decoding, and stacking weights maps to each subjects inflated cortical surface. 
  - `plot_learning_curves`: given the results of `bin_compute_confidence.py` (and `analysisRL.m` for long-term 4AFC trials), plots the accuracy over time learning curves shown in Fig. 1E and S1
  - `plot_selectivity.py`: given the results of  `compute_selectivity.py`, plots bar plots of selectivity in ROI shown in Fig. 4C,D
  - `plot_surface_comparison.py`: given the results of `surface_comparison.py`, plots the posterior to anterior progression of the passive task biases shown in Fig. 4E
  - `plot_color_biased_region.py`: given the results of `color_biased_regions.py`, plots line plots of colorshown in Fig. 4F
- data: an empty directory where you are expected to place project data from data repository in order to easily run analysis code on it.
- results: an empty directory that will save intermediate results from analysis scripts before visualization (e.g. csv files of accuracy in each ROI for each CV fold)
- figures: directory to hold visualization scripts svg outputs. 

## Software Used
- only tested on Ubuntu 22.04 LTS
- python==3.12.4
- numpy==1.26.4
- scipy==1.12.0
- pandas==2.2.2
- matplotlib==3.8.4
- scikit-learn==1.5.1
- scikit-image==0.24.0
- nibabel==5.2.1
- nilearn==0.8.2
- torch==2.7.0
- neurotools==0.0.2

Fitting the searchlight model in a reasonable amount of time requires a nvidia GPU. We used a GTX 4090TI. 
All of these python packages should be installed in you local environment.

## Loading data
Project data can be downloaded from figshare: https://figshare.com/projects/Datasets_for_The_representation_of_object_concepts_across_the_brain_/255242

This data can be put into the proper directory structure by downloading the zip file for each separate experiment into 
this root of this repository, and then running the `collect_from_figshare.sh` shell script to generate the `data` directory.
The `searchlight_runner.py` script should then be able to properly find the data when run with the repository root as the
working directory. 

All other scripts should also be able to work with this directory structure. 


  
