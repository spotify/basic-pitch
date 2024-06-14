# Data / Training
The code and scripts in this section deal with training basic pitch on your own. Scripts in the `datasets` folder allow one to download and process a selection of the datasets used to train the original model. Each of these download scripts has the following keyword arguments:
* **--source**: Source directory to download raw data to. It defaults to `$HOME/mir_datasets/{dataset_name}`
* **--destination**: Directory to write processed data to. It defaults to `$HOME/data/basic_pitch/{dataset_name}`.
* **--runner**: The method used to run the Beam Pipeline for processing the dataset. Options include `DirectRunner`, running directly in the code process running the pipeline, `PortableRunner`, which can be used to run the pipeline in a docker container locally, and `DataflowRunner`, which can be used to run the pipeline in a docker container on Dataflow. 
* **--timestamped**: If passed, the dataset will be put into a timestamp directory instead of 'splits'.
* **--batch-size**: Number of examples per tfrecord when partitioning the dataset.
* **--sdk_container_image**: The Docker container image used to process the data if using `PortableRunner` or `DirectRunner` .
* **--job_endpoint**: the endpoint where the job is running. It defaults to `embed` which works for `PortableRunner`. 

Additional arguments that work with Beam in general can be used as well, and will be passed along and used by the pipeline. If using `DataflowRunner`, you will be required to pass `--temp_location={Path to GCS Bucket}`, `--staging_location={Path to GCS Bucket}`, `--project={Name of GCS Project}` and `--region={GCS region}`. 
