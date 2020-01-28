# Tidepool Donor Data Pipeline

This repository will contain the code and resources used for processing data from [Tidepool's Big Data Donation Project](https://www.tidepool.org/bigdata).

**Note:** All code is currently located as the *bigdata-processing-pipeline project* within [tidepool-org/data-analytics](https://github.com/tidepool-org/data-analytics/tree/master/projects/bigdata-processing-pipeline) and will soon be migrated into this repository for further development.

### What is it?

------

The **Donor Data Pipeline** is the set of tools which help transform Tidepool's donated diabetes device data into clean and de-identified datasets ready for analysis.

This pipeline includes code which:

* Collects datasets through Tidepool's API calls
* Cleans and corrects values, timestamps, and data formats
* Creates estimates of local time from device data
* Anonymizes datasets
* Summarizes dataset content and quality

To learn more about the donated data, see Tidepool's list of [supported devices](https://www.tidepool.org/users/devices) and the [data model for diabetes device data types](http://developer.tidepool.org/data-model/device-data/types/).

### Getting Started

------

Check out the [src README](./src) and Quick Start Guide to get orientated with the pipeline code and how to use it. 

### Getting Help

------

Tidepool has a [Public Slack](https://tidepoolorg.slack.com/) and you can reach the Tidepool Data Science Team within the #data-analytics channel. You may also send in your questions to bigdata@tidepool.org.

### TODO:

------

- [x] Add README
- [x] Migrate existing code from bigdata-processing-pipeline project in tidepool-org/data-analytics
- [x] Update new Getting Started README link