
#  Build

-  First clone `BitNet` repository:

	-  `./clone_bitnet.sh`

-  Build [`BitNet`](https://github.com/microsoft/BitNet) project and setup env:

	-  `cd BitNet`

	-  `conda create -n bitnet-cpp python=3.9`

	-  `conda activate bitnet-cpp`

	-  `pip install -r requirements.txt`

	-  `python setup_env.py --hf-repo tiiuae/Falcon3-7B-Instruct-1.58bit -q i2_s`

		-  This step takes a few minutes. Check the [original repository](https://github.com/microsoft/BitNet) in case of any errors.

-  Build current project:

	-  `./build.sh`

		-  This script is inside 'bitnetcpp' dirctory.

  

#  Run Experiment

-  Run the command `./experiments.sh`. This will print the time reports in a file: `time_report.txt`.
