# How to install the environment and run the exercises

For those attending the workshop and who are based in Finland, we provide access to the pre-installed enviournment via the https://noppe.csc.fi interface. 

## How to use Noppe

After logging in with your HAKA Finnish university account, you will see various jupyter lab instances that can be started with a big round "on/off" button. The instance we use for the course is not there in the standard list. To add you need to click on "Join Workspace" and insert the code that was given to you via email.

## How to run the exercises without Noppe

If you want to run the notebooks in this repository without using noppe (for example because you are not affiliated with any Finnish research organisation) you can use Apptainer/Singularity or Docker to pull the container that contains all the dependencies. Apptainer/Singularity works well on Linux, especially in shared systems like HPC systems. Docker works with all major OSs, but it requires admin permissions.

### Running with Apptainer

First, create a new working folder for the exercises and enter it:

```bash
mkdir ml-reproducibility-workshop
cd ml-reproducibility-workshop
```

Pull the Apptainer image:

```bash
apptainer pull ml-reproducibility_latest.sif docker://harbor.cs.aalto.fi/aaltorse-public/ml-reproducibility:latest
```

Clone the exercise repository:

```bash
git clone https://github.com/AaltoSciComp/ml-reproducibility-examples
```

You should now have the Apptainer image and the exercise folder in the same directory.

Start Jupyter Lab inside the Apptainer container:

```bash
apptainer exec --bind "$PWD":/workspace ml-reproducibility_latest.sif \
  bash -lc 'cd /workspace && jupyter lab --no-browser --ip=127.0.0.1 --port=8888'
```

This command makes your current folder available inside the container as `/workspace`, so the cloned exercise repository will be visible in Jupyter Lab.

After running the command, Jupyter will print a link that looks something like this:

```text
http://127.0.0.1:8888/lab?token=...
```

Copy and paste the full link, including the token, into your web browser.

In Jupyter Lab, open the folder:

```text
ml-reproducibility-examples
```

Then open one of the exercise notebooks and try running it.

### Running with Docker


First, create a new working folder for the exercises and enter it:

```bash
mkdir ml-reproducibility-workshop
cd ml-reproducibility-workshop
```

Pull the Docker image:

```bash
docker pull harbor.cs.aalto.fi/aaltorse-public/ml-reproducibility:latest
```

Clone the exercise repository:

```bash
git clone https://github.com/AaltoSciComp/ml-reproducibility-examples
```

You should now have the exercise folder in your current directory.

Start Jupyter Lab inside the Docker container:

```bash
docker run --rm -it \
  -p 8888:8888 \
  -v "$PWD":/workspace \
  -w /workspace \
  harbor.cs.aalto.fi/aaltorse-public/ml-reproducibility:latest \
  bash -lc 'jupyter lab --no-browser --ip=0.0.0.0 --port=8888 --allow-root'
```

This command makes your current folder available inside the container as `/workspace`, so the cloned exercise repository will be visible in Jupyter Lab.

After running the command, Jupyter will print a link that looks something like this:

```text
http://127.0.0.1:8888/lab?token=...
```

Copy and paste the full link, including the token, into your web browser.

In Jupyter Lab, open the folder:

```text
ml-reproducibility-examples
```

Then open one of the exercise notebooks and try running it.

### Build your own Python environemnt using conda/mamba

You can also re-build the same Python environment using conda / mamba. Follow the same instructions as the conda installation for our CodeRefinery workshop at: https://coderefinery.github.io/installation/conda/ but, in this case, please use the environment.yml available in this repository.

The command to set up the environment will then be:

```
mamba env create -n ml-reproducibility -f https://raw.githubusercontent.com/AaltoSciComp/ml-reproducibility-examples/refs/heads/main/environment.yml
```
