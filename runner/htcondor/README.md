
# multicore/thread on HTCondor (WiP)


## miminal component

- the flow: `example_flowbdt.py`
- the script to setup how to run: `example_script.py`
- the submit file to configure condor: `submit.sub`


```bash
# can either run locally
source example_script.sh

# or send the job to the grid
condor_submit submit.sub
condor_q

```
