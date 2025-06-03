# ColorShapeConcepts

Code needed for the paper "The representation of object concepts across the brain". Includes:
- Code for running searchlight cross decoding and other algorithms
- Generating rough versions of figures in the paper

Much of this code is dependent on the `neurotools` library ([github](https://github.com/spencer-loggia/neurotools)), which contains core algorithms and utility functions. 

## Structure:
- analyze
  - `searchlight_runner.py`: contructs the LSDM (or standard searchlight), connects the LSDM to the fMRI dataloader that feed trial data and labels, runs cross validation procedure (identity and cross decoding), saves results to csv
  - `simulation_runner.py`: contrusts an LSDM and standard searchlight, connects to the simulation dataloader, for different input set sizes preforms cross validated indentity and cross decoding of the simulated data for both models and saves the results to csv.
  - `compute_embeddings.py`: Given a trained LSDM output directory (containing binaries of trained models on each CV fold), computes the representational dissimilarity matrices for each ROI in the LSDM latent space (see paper methods) and saves them as npy files with a cssv key.
- bin
  - `dataloader.py` defines a class that is designed to load fMRI data in parallel from the data directory and serve batches of it to the searchlight runner.
  - `searchlight_dataloader.py` same as above, but generates simulated data as described in the methods.
  - `plotter.py` defines scripts used for generating plots of data.
- visualize
  - `decoding_plots_from_results.py` given the path(s) to output model diretory(s) (as producedby searchlight runner and containing 'results.csv') bootstrap the results, produce plots of decdoing performance over ROIs
  - `geometry_plots_from_results.py` given the path(s) to output model diretory(s) that have been processed by the `compute_embeddings.py` script and thus contain 'rdm_key.csv', produce MDS plots and bootstrapped plots of correlation with color space geometry.
  - `create_searchlight_simulation_plots.py` given the results of the `simulation_runner.py` script, create the simulation decoding plots shown in Fig. S3.
  - `decoding_results_to_surface.py` script to project the identity, cross decoding, and stacking weights maps to each subjects inflated cortical surface. 
- data: an empty directory where you are expected to place project data from data repository in order to easily run analysis code on it.
- results: an empty directory that will save intermediate results from analysis scripts before visualization (e.g. csv files of accuracy in each ROI for each CV fold)
- figures: directory to hold visualization scripts svg outputs. 

  
