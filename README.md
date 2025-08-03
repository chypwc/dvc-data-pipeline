# DVC Pipeline - IMDB Sentiment Analysis

## Dataset

IMDB Dataset of 50K Movie Reviews  
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

## Pipeline Stages

1. **prepare_data** - Split dataset into train/test sets
2. **make_features** - Create TF-IDF features from text data
3. **train** - Train logistic regression model
4. **evaluate** - Generate evaluation metrics

## Usage

```bash
git init
dvc init

# S3 config
dvc remote add -d s3 s3://<bucket>/<key_prefix>/
dvc remote modify --local s3 configpath '~/.aws/config'
```

```bash
# Run entire pipeline
dvc repro

# View pipeline DAG
dvc dag

# Save data
dvc push
```

### Pipeline DAG

```bash
  +--------------+
                          | prepare_data |
                        **+--------------+***
                    ****          *          ****
                ****              *              ****
             ***                  *                  ****
+---------------+                 *                      ***
| make_features |                 *                        *
+---------------+***              *                        *
        *           ****          *                        *
        *               ****      *                        *
        *                   ***   *                        *
        ***                   +-------+                  ***
           ****               | train |              ****
               *****          +-------+          ****
                    ****          *          ****
                        ****      *      ****
                            ***   *   ***
                            +----------+
```

## Configuration

Edit `params.yaml` to modify:

- Data split ratio
- Feature extraction method
- Model hyperparameters
- Evaluation metrics
