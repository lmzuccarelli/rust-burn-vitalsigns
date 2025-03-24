# Overview

Using burn libraries with cuda (gpu) to train and infer a vitals signs data set 

## Usage

**N.B.**  The data for training, validating and testing is proprietary and has not 
been included in this repo for obvious reasons.

The DataLoader will look for files named

- data/vital-signs-100.csv (used for training)
- data/vital-signs-validate.csv (used for validating)
- data/vital-sign-test.csv (used for testing)

Change the directories and file names before compiling.

clone the repo

```
cd rust-burn-vitalsigns

# build 

cargo build --release

#execute

./target/release/rust-burn-vitalsigns train
```
