# Donor Data Pipeline Source Code README

Welcome to the home of the source code and contents supporting the Tidepool Donor Data Pipeline! 

Below you will find:

* **Pipeline Flow** - An overview of the order, structure, and purpose of each pipeline module
* **Batch Processing Tools** - A review of different useful batch processing tools that can be used with the pipeline and its modules
* **Quick Start Guide** - A quick setup guide and examples to use the pipeline and batch tools
* **Pipeline Output** - Information about the output folders, files, and contents created by the pipeline
* **TODO** - Upcoming improvements and features
  


### Pipeline Flow

------

Below is a list of all modules used in the main pipeline in order of execution.

* **donor_data_pipeline.py**
  * This is the main wrapper for the entire donor data pipeline. All other modules are imported and executed from here. It acts as a data manager, feeding data into and out of each module.
* **environmentalVariables.py**
  * Loads environmental variables from an .env file.
* **accept_new_donors_and_get_donor_list.py**
  * (optional) - Logs into each Tidepool Big Data Donor account and accepts all pending donors who have opted-into sharing their data with the TBDDP. Returns a list of all donor userids. 
* **get_single_donor_metadata.py**
  * Logs into a big data or user account and returns all user metadata
* **get_single_tidepool_dataset.py**
  * Logs into a big data or user account and returns all user device data
* **estimate_local_time_v0_4.py**
  * An algorithm to estimate the local time for each device record within the donor data. Useful for finding duplicates, correcting timezone shifts, and other device time errors.
* **vector_qualify.py** 
  * Analyzes daily data content and quality to determine the different types of data available and their duration. Useful for separating datasets into sensor augmented pump therapy (SAP), hybrid-closed loop (HCL), and Physical Activity (PA).
* **anonymize_and_export.py** 
  * Cleans, flattens, and transforms, and anonymizes the data into its final form
* **split_test_train_datasets.py**
  * If separating a dataset into training and testing sets, this module finds the optimal split date given the desired amount of days in the test set.
* **anonymized_stats_qa.py**
  * Collects a wide range of statistics on the anonymized datasets for later quality assurance checks
* **dataset_summary_viz.py** 
  * Creates a visual time-series summary of the anonymized dataset contents
* **local_time_summary_viz.py** 
  * Creates a visual time-series summary of the estimated local time algorithm timezones applies throughout the dataset

### Batch Processing Tools

------

The Tidepool Data Science Team uses the donor data pipeline and its modules to process information across many donor datasets. Batch processing tools are used to do this efficiently often with asynchronous parallelization. 

Below are descriptions of the different tools in their typical order of execution. Note that all batch filesnames are dash-separated unlike the underscore_separated importable modules.

* **batch-get-donor-data.py**
  * Runs the get_single_tidepool_dataset.get_and_return_dataset() function to download all donor data from a list of userids and their associated donor group accounts. This file can be created using accept_new_donors_and_get_donor_list.py.  
* **batch-get-metadata.py**
  * Runs get_single_donor_metadata.py on a list of userids and their associated donor group accounts, then and combines all of their metadata into a single file.
* **batch-vector-qualify.py**
  - Runs the vector_qualify.get_vector_summary() function on a local folder of datasets and creates a combined vector qualification .csv. Useful for finding and selecting donor datasets of specific sizes, quality, and data types.
* **batch-pipeline.py**
  - Runs donor_data_pipeline.py and exports pipeline and datasets summaries across all donors processed. Requires output from the previous three batch processing scripts.
* **batch-get-unique-dataframes.py**
  * Creates a summary of all unique value-lengths for every column in every dataset from a folder of datasets. An optional but useful tool for getting an overview of every type of unique data type and format.

### Quick Start Guide

------

Here's a quick setup guide to get you started.

**Setup Virtual Environment** 

The Donor Data Pipeline is run within an anaconda virtual environment. [Anaconda](https://www.anaconda.com/distribution/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (recommended) can be used to install the `conda` command line tool.

1. In a teriminal, navigate to this repository's root folder where the `environmental.yml` file is located.
2. Run `conda env create`. This will download all of the pipeline's package dependencies and install them in a virtual environment named **tbddp** (**T**idepool **B**ig **D**ata **D**onation **P**roject)
3. Run `conda activate tbddp` to activate the environment and `conda deactivate` at anytime to exit. Note: You may optionally use the virtualenv `source` command instead of `conda` to activate and deactivate the environment.

**Add Environmental File Dependency (Tidepool Employees Only)**

The donor data pipeline depends on a private `.env` environmental file to access donor data via the Tidepool API. You will need to add this file to the `src` directory before proceeding. Reach out to the Tidepool Data Science Team for information on getting access to this file.

**Process A Single Tidepool Dataset**

1. In the terminal, navigate to the **src** directory
2. Run `python donor_data_pipeline.py`
3. When prompted, enter the user id of the Tidepool account you want to have processed
  

Now just wait for the pipeline to finish processing and that's it! Your processed data will be located in a newly created pipeline export folder. See the next section for more details.

**Batch Download Data**

1. In the terminal, navigate to the **src** directory
2. Run `python accept_new_donors_and_get_donor_list.py` - this creates the file: data/PHI-YYYY-MM-DD-donor-data/PHI-YYYY-MM-DD-uniqueDonorList.csv
3. Open `batch-get-metadata.py` and insert the unique donor list file path into the `phi_donor_list` variable's read_csv function
4. Run `python batch-get-metadata.py` - this creates the file `PHI-batch-metadata-YYYY-MM-DD.csv`
5. Open `batch-get-donor-data.py` and insert the metadata filepath into the variable `chosen_donors_file`
6. Run `python batch-get-donor-data.py` and wait for all data to be downloaded into the new folder PHI-YYYY-MM-DD-csvData. Data is saved in gzip compressed format by default.


### Pipeline Output

------

The folder structure, contents, and explanations of the pipeline output are as follows. Folders are in bold.

* **PHI-pipeline-export-\<timestamp>**
  * **PHI-csvData**
    * Contains saved raw .csv donor account data if the `-save_new_data` / `saveDataDownload` parameter is set to true 
  * **PHI-pipeline-results**
    * Contains pickled .data objects with an array of pipeline results and metadata. Primarily used by the batch-pipeline.py tool to combine all results into a single summary.
  * **QA**
    * **PHI-LTE-cDays**
      * Contains .csvs from the estimate local time module with a continuous day series and the algorithm's metadata for each day
    * **PHI-vector-qualified-days**
      * Contains .csvs from the vector qualify module with a continuous day series of what each day's data qualifies as
    * **viz-QA**
      * **\<test/train>-dataset-summary-vizQA**
        * Contains .png image visualizations of the test/train dataset content types available over time such as
      * **\<test/train>-vector-qualify-vizQA**
        * Contains .png image visualizations of the test/train dataset qualified segments generated by the vector qualify module. A visual representation similar to the data found in PHI-vector-qualified-days folder.
      * **\<test/train>-local-time-vizQA**
        * Contains .png image visualizations of the test/train dataset estimated local timezones throughout the dataset
      * **plotly-dropped-data-vizQA**
        * Contains .html interactive plotly visualizations of the dataset after processing to show what data has been dropped. Useful for verifying deduplication and noisiness of the data.
  * **train**
    * **train-data**
      * Contains .csvs of the full processed training data 
  * **test**
    * **test-data**
      * Contains .csvs of the fully processed testing data



### TODO:

------

- [ ] Add information on the different pipeline arguments
- [ ] Add information on batch processing setup
- [ ] Add section for expanding pipeline with new modules
- [ ] Change batch data downloading to take in arguments instead of manually changing variables.
