Identifying Humpback Whale Calls
----

Exploratory and processing pipeline for applying a
deep learning model[^1] that identified humpback
whale calls to a set of continuous audio data
collected off the coast of central California[^2].
The output of the pipeline produces a series of
hourly csv files with prediction probabilities
for each 1s bin of recording.

[^1]: A. Allen et al., "A convolutional neural network for automated detection of humpback whale song in a diverse, long-term passive acoustic dataset", Front. Mar. Sci., 2021, doi: 10.3389/fmars.2021.607321.

[^2]: Pacific Ocean Sound Recordings from https://registry.opendata.aws/pacific-sound.

## Setup for redun pipeline

#### Local development
You can run the redun pipeline locally if you
change the executor to "local".

Clone the repo then setup an virtual env for local
testing:
```bash
python3 -m venv venv
source venv bin activate
pip install -r requirements.txt
```
Download the model at https://tfhub.dev/google/humpback_whale/1
and unpack into the same directory as the code
and rename the folder `humpback_model_cache`.

#### AWS
You'll need an AWS account and willingness to spend
about $0.50 per day analyzed on EC2 bills if you want to run 
the entire pipeline.

Install the aws-cli (v2) and configure
with your credentials and a role that can
create s3 buckets, ECR repos, and all things ECS.

(TODO: better instructions)

Setup an EC2 compute environment for AWS Batch.
I configured mine with 32 vCPUs which should be
around $1.20 an hour.
Configure a job queue that uses the compute environment.
Configure an s3 bucket for results and redun logs. 

#### Docker for running in AWS Batch
Build the docker image and deploy to your own ECR:
```bash
make login
make create-repo
make build
make push
```
Modify your redun.ini in `.redun/`
```bash
[limits]
batch = 8  # configure as you see fit

[executors.batch]
type = aws_batch
image = <your account number>.dkr.ecr.us-west-2.amazonaws.com/humpack_pipeline:latest
queue = <queue name>
s3_scratch = s3://<some bucket>/redun/scratch/

aws_region = us-west-2
role = <arn to role with ECS Task and read/write to s3 bucket
debug = False
code_excludes = venv/*
vcpus = 4
memory = 8
```

## Running
For a single day:
```
redun run humpback_pipeline.py run_for_date \
--date '2021-01-01' \
--output_dir 's3://something'  # no trailing slash
```

With 32 vCPU compute environment and 8 parallel tasks
using 4 vCPU each, this took ~20min (way faster
that the 2 hr running the pipeline on my dual core i5
laptop).

For a range (TODO):
```
redun run humpback_pipline.py run_for_range \
--start '2021-01-01' \
--stop '2022-01-01' \
--output_dir 's3://something'  # no trailing slash
```


## Notes

This is a poor use of redun since we're not
really leveraging reuse and we're not really
chunking up work.  But, that's because I didn't
want to create a bunch of intermediate files
(like chunked wav or spectrograms) that would
be kept around in s3 for a long time.  Instead,
we're using redun mostly to coordinate running
chunks of model fitting in AWS Batch.

To do:
* Run pipeline on a significant chunk of time
(only tested a few days so far)
* Median filter and threshold scores
* Do some interesting analysis on the scored data