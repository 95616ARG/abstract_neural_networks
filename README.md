# Abstract Neural Networks
This repository contains code related to the SAS 2020 paper _Abstract Neural
Networks_ by Matthew Sotoudeh and Aditya V. Thakur.

## Using
#### Dependencies
This code assumes reasonably up-to-date versions of `python3`, `numpy`, and
`pytest` in order to run correctly. If you already have Python installed, you
can install the other two dependencies like so:
```bash
python3 -m pip install -r requirements.txt
```

#### Usage
The file `abstract.py` exposes a method `abstract_layer_wise` which corresponds
to Algorithm 3 in our paper. An example of its uses is provided in the file
`test_abstract_intervals.py`, corresponding to Example 6 (Section 5.1) in our
paper. To run it, and any other test cases, you can use:
```bash
python3 -m pytest *.py
```
in this directory.

#### Bazel Usage
Optionally, we support Bazel for reproducible runs and testing. After setting
up [bazel_python](https://github.com/95616ARG/bazel_python), you can run
```bash
bazel test //...
```
to run all test cases, then
```bash
bazel run coverage_report
```
to produce an HTML coverage report in a new `htmlcov` directory.

## Citing
```bibtex
@inproceedings{anns:sas20,
    author = {Sotoudeh, Matthew and Thakur, Aditya V.},
    title = {Abstract Neural Networks},
    booktitle = {27th Static Analysis Symposium (SAS)},
    year = {2020},
    note = {To appear}
}
```
