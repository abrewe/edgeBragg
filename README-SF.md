# Testing with pvapy streaming framework 

## Required software 

PvaPy installation:

```sh
$ conda install -c sveseli pvapy
```

## Examples

### Single Consumer

Terminal 1 (generate images):

```sh
$ pvapy-ad-sim-server \
    -cn pvapy:image \
    -nx 128 -ny 128 -dt int16 \
    -fps 10 -rp 10 -rt 60
```

Terminal 2 (Bragg NN):

```sh
$ export PYTHONPATH=/path/to/edgeBragg
$ pvapy-hpc-consumer \
    --input-channel=pvapy:image \
    --output-channel=bragg:*:output \
    --control-channel=bragg:*:control \
    --status-channel=bragg:*:status \
    --processor-file=/path/to/edgeBragg/braggNNInferImageProcessor.py \
    --processor-class=BraggNNInferImageProcessor \
    --processor-args='{"configFile" : "/path/to/edgeBragg/config/sim.sf.yaml"}' \
    --report-period=10 \
    --log-level=DEBUG
```

### Multiple Consumers

Terminal 1 (generate images):

```sh
$ pvapy-ad-sim-server \
    -cn pvapy:image \
    -nx 128 -ny 128 -dt int16 \
    -fps 10 -rp 10 -rt 60
```

Terminal 2 (Bragg NN):

```sh
$ export PYTHONPATH=/path/to/edgeBragg
$ pvapy-hpc-consumer \
    --input-channel=pvapy:image \
    --output-channel=bragg:*:output \
    --control-channel=bragg:*:control \
    --status-channel=bragg:*:status \
    --processor-file=/path/to/edgeBragg/braggNNInferImageProcessor.py \
    --processor-class=BraggNNInferImageProcessor \
    --processor-args='{"configFile" : "/path/to/edgeBragg/config/sim.sf.yaml"}' \
    --n-consumers 4 \
    --distributor-updates 1 \
    --report-period=10 \
    --log-level=DEBUG
```

In order to get consumer status at runtime use the following command:

```sh
$ pvget bragg:1:status
```

